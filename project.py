import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

def get_retriever(vectorstore, selected_doc="All Documents"):
    if selected_doc != "All Documents":
        return vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_doc}, "k": 4})
    else:
        return vectorstore.as_retriever(search_kwargs={"k": 8})

def get_pdf_documents(pdf_docs):
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                annotated_content = f"---Document: {pdf.name} (Page {page_num + 1}) ---\n{text}"
                doc = Document(page_content=annotated_content, metadata={"source": pdf.name, "page_num": page_num + 1})
                documents.append(doc)
    return documents

'''
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
'''

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        #separator="\n",
        chunk_size=3500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
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
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    retriever = get_retriever(vectorstore, selected_doc)

    #if selected_doc != "All Documents":
    #    retriever = vectorstore.as_retriever(search_kwargs={"filter": {"source": selected_doc}, "k": 4})
    #else:
    #    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    custom_template = """
    You are a helpful assistant answering questions using the retrieved context from uploaded PDF documents. Each chunk includes
    its document name header(e.g., '---Document: <filename> (Page <num>) ---').

    Rules:
    1. When asked about a specific person or document, list ALL relevent entries (such as ALL work experiences, jobs, or projects)
    found for that document. Do not stop after just one.
    2. DO NOT mix or combine information from different documents.
    3. If the answer cannot be found in the context, say that you do not have enough information.

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
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        # return_source_documents=True
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

if __name__ == '__main__':
    main()