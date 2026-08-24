# ChatWithPDFs

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs and ask questions about their content.

## How It Works

```text
PDFs → Text Extraction → Chunking → Embeddings → FAISS
                                                    ↓
Question → Retrieve Relevant Chunks → Groq LLM → Answer
```

## Features

* Upload and process multiple PDFs
* Text extraction and chunking
* Local embeddings using `all-MiniLM-L6-v2`
* FAISS-based vector search
* Conversational Q&A with Groq LLM
* Streamlit chat interface

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
