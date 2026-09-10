# ChatWithPDFs

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs and ask questions about their content.

## How It Works

```text
PDFs → Page-Level Metadata Extraction → Context Chunking → Embeddings → FAISS Index
                                                                             ↓
Question (+ Dynamic Filter) → Targeted Retrieval (Top-k) → Groq LLM → Grounded Answer
```

## Features

* **Upload & Process Multiple PDFs**: Extracts text per page with attached metadata (`source`, `page_num`).
* **Dynamic Metadata Filtering**: Filter queries to a specific document or search globally across all uploaded PDFs via the sidebar.
* **Anti-Hallucination Prompting**: Annotated document chunk headers and strict QA prompting prevent cross-document contamination.
* **Fast Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` cached with `@st.cache_resource` for offline, sub-second vector generation.
* **FAISS Vector Search**: Fast similarity search with customizable top-$k$ retrieval parameters.
* **Conversational Memory**: Multi-turn chat memory (`ConversationBufferMemory`) retains context across follow-up queries.
* **Modern Streamlit Interface**: Clean, responsive UI with automatic chat input clearing (`st.chat_input`).

## Tech Stack

Python · Streamlit · LangChain · FAISS · Hugging Face · Groq · PyPDF2

## Project Structure

```text
ChatWithPDF/
├── app.py
├── htmlTemplates.py
├── requirements.txt
└── README.md
```

## Setup & Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key to `.env`:**
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```
