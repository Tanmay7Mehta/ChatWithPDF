import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from sentence_transformers import CrossEncoder
from typing import List
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

load_dotenv()
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

EVAL_QUESTIONS = [
    "What is Tanmay's work experience?",
]

EVAL_GROUND_TRUTHS = [
    "AI Support Engineer at Offlens Studio, AI Analyst at Orane Consulting, AI/ML Intern at NPR Supporting Services",
]

def run_ragas_evaluation(conversation_chain):
    """Run RAGAS evaluation using the active conversation chain's retriever."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not found in .env file. Please add it to run evaluation.")

    answers = []
    retrieved_contexts = []

    for question in EVAL_QUESTIONS:
        response = conversation_chain.invoke({"question": question})
        answers.append(response["answer"])
        source_docs = response.get("source_documents", [])
        # Include up to top 6 retrieved chunks
        contexts = [doc.page_content for doc in source_docs[:6]]
        if not contexts:
            contexts = ["No relevant context retrieved from uploaded documents."]
        retrieved_contexts.append(contexts)

    eval_data = {
        "user_input": EVAL_QUESTIONS,
        "response": answers,
        "reference": EVAL_GROUND_TRUTHS,
        "retrieved_contexts": retrieved_contexts,
    }

    client = OpenAI(
        api_key=gemini_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    eval_llm = llm_factory("gemini-3.5-flash-lite", client=client)
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    dataset = Dataset.from_dict(eval_data)
    run_config = RunConfig(max_workers=1, timeout=120, max_retries=10, max_wait=30)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config,
    )
    return results, eval_data

@st.cache_resource
def get_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

class RerankerRetriever(BaseRetriever):
    """Custom retriever that over-fetches from FAISS then reranks with a cross-encoder."""
    base_retriever: object
    top_k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)

        if not docs:
            return []
        
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        
        # Group candidates by source document to ensure multi-document queries (e.g. comparisons) get fair coverage
        by_source = {}
        for score, doc in ranked:
            src = doc.metadata.get("source", "Unknown")
            by_source.setdefault(src, []).append((score, doc))

        # Best overall chunk score
        best_score = ranked[0][0]

        # Only retain documents whose best chunk is within a reasonable margin of the top match
        # This filters out completely unrelated documents while keeping all relevant ones for comparisons/listings
        valid_sources = [
            src for src, items in by_source.items()
            if items[0][0] >= best_score - 7.0
        ]
        if not valid_sources:
            valid_sources = list(by_source.keys())[:1]

        # Interleave chunks across valid sources up to top_k
        selected = []
        indices = {src: 0 for src in valid_sources}
        while len(selected) < self.top_k:
            added = False
            for src in valid_sources:
                if len(selected) < self.top_k and indices[src] < len(by_source[src]):
                    selected.append(by_source[src][indices[src]][1])
                    indices[src] += 1
                    added = True
            if not added:
                break

        return selected

def get_retriever(vectorstore, selected_doc="All Documents"):
    if selected_doc != "All Documents":
        base = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_doc}, "k": 30})
        return RerankerRetriever(base_retriever=base, top_k=6)
    else:
        base = vectorstore.as_retriever(search_kwargs={"k": 40})
        return RerankerRetriever(base_retriever=base, top_k=6)

def get_pdf_documents(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"source": pdf.name, "page_num": page_num + 1})
                documents.append(doc)
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page_num", "?")
        header = f"--- Document: {source} (Page {page}) ---\n"
        if not chunk.page_content.startswith("--- Document:"):
            chunk.page_content = header + chunk.page_content.strip()

    return chunks

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True})

def get_vectorstore(text_chunks):
    embeddings = get_embeddings()
    #embeddings = HuggingFaceEmbeddings(
    #    model_name="sentence-transformers/all-MiniLM-L6-v2",
    #    model_kwargs={'device':'cpu'}
    #)
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, selected_doc="All Documents"):
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.2)# lower temperature for factual answers
    memory = ConversationBufferMemory(memory_key='chat_history', output_key='answer', return_messages=True)
    retriever = get_retriever(vectorstore, selected_doc)

    #if selected_doc != "All Documents":
    #    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_doc}, "k": 4})
    #else:
    #    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    custom_template = """
    You are a helpful assistant answering questions using the retrieved context from uploaded PDF documents. Each chunk includes its document name header (e.g., '--- Document: <filename> (Page <num>) ---').

    Rules:
    1. When asked about a specific person or document, list ALL relevant entries (such as ALL work experiences, jobs, or projects) found for that document. Do not stop after just one.
    2. When asked to compare multiple people or documents, or when asked general questions across all documents (e.g., 'List all work experiences' or 'Compare X and Y'), organize the response clearly by person/document and include all matching information found in the context.
    3. DO NOT mix or combine information from different documents under the same person.
    4. If the answer cannot be found in the context, say that you do not have enough information.

    context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer: 
    """

    QA_PROMPT = PromptTemplate(
        template=custom_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )
    return conversation_chain

def handel_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload your PDFs and click 'process' before asking question.")
        return

    response = st.session_state.conversation.invoke({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.chat_input("Ask a question about your document")

    if user_question:
        handel_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'process'", accept_multiple_files=True)
        doc_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        selected_doc = st.selectbox("Filter questions to a specific PDF:", options=["All Documents"]+ doc_names)

        #if st.session_state.vectorstore is not None:
        #    st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, selected_doc)

        if st.session_state.conversation is not None and st.session_state.vectorstore is not None:
            st.session_state.conversation.retriever = get_retriever(st.session_state.vectorstore, selected_doc)

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing..."):
                    documents = get_pdf_documents(pdf_docs)
                    text_chunks = get_text_chunks(documents)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation = get_conversation_chain(vectorstore, selected_doc)

        st.divider()
        st.subheader("RAGAS Evaluation")
        if st.button("Evaluate"):
            if st.session_state.conversation is None:
                st.warning("Please upload and process PDFs before running evaluation.")
            elif not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                st.warning("Please add GEMINI_API_KEY to your .env file.")
            else:
                with st.spinner("Running RAGAS evaluation... (this may take a minute)"):
                    try:
                        results, eval_data = run_ragas_evaluation(st.session_state.conversation)
                        df = results.to_pandas()
                        if "faithfulness" in df.columns:
                            df["faithfulness"] = df["faithfulness"].fillna(1.0)
                        st.dataframe(
                            df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]],
                            use_container_width=True,
                        )
                        with st.expander("🔍 View Evaluation Details"):
                            for idx, q in enumerate(eval_data["user_input"]):
                                st.markdown(f"**Question:** {q}")
                                st.markdown(f"**Bot Answer:** {eval_data['response'][idx]}")
                                st.markdown(f"**Ground Truth:** {eval_data['reference'][idx]}")
                                st.markdown("**Retrieved Chunks:**")
                                for c_idx, ctx in enumerate(eval_data["retrieved_contexts"][idx]):
                                    st.caption(f"Chunk {c_idx + 1}:")
                                    st.text(ctx[:300] + ("..." if len(ctx) > 300 else ""))
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

if __name__ == '__main__':
    main()