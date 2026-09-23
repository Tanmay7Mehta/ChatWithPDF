# ChatWithPDFs

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDFs and ask questions about their content, with source citations and automated RAG evaluation.

## How It Works

```text
PDFs
  ↓
Text & Page Metadata Extraction
  ↓
Recursive Chunking (with Document Headers)
  ↓
Hugging Face Embeddings
  ↓
FAISS Retrieval (Top-k Candidates)
  ↓
Cross-Encoder Reranking
  ↓
Groq LLM
  ↓
Answer + Source Citations
```

## Features

* **Upload & Process Multiple PDFs**: Extracts text per page with attached metadata (`source`, `page_num`).
* **Source Citations**: Displays source documents, page numbers, and retrieved text snippets directly under each answer.
* **Two-Stage Retrieval**: Over-fetches candidate chunks using FAISS and refines them with Cross-Encoder reranking.
* **Multi-Document Support**: Supports cross-document comparisons and lists while maintaining source context.
* **Document Filtering**: Query a specific uploaded document or search across all documents.
* **Conversational Memory**: Retains multi-turn chat history for follow-up questions.
* **RAGAS Evaluation**: Built-in evaluation directly in the sidebar to benchmark Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

## Tech Stack

Python · Streamlit · LangChain · FAISS · Hugging Face · CrossEncoder · Groq · RAGAS · PyPDF2

## Project Structure

```text
ChatWithPDF/
├── project.py
├── htmlTemplates.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanmay7Mehta/ChatWithPDF.git
   cd ChatWithPDF
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Create a `.env` file in the root directory (see `.env.example`):
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```
   *(Gemini API key is free and used for running RAGAS evaluation)*

4. **Run the application:**
   ```bash
   streamlit run project.py
   ```

## Evaluation

The application includes an optional RAGAS evaluation workflow in the sidebar to evaluate the RAG pipeline using:

* **Faithfulness**: Factual consistency against retrieved context.
* **Answer Relevancy**: Pertinence of the generated response to the query.
* **Context Precision**: Signal-to-noise ratio in retrieved context.
* **Context Recall**: Retrieval completeness against reference ground truth.
