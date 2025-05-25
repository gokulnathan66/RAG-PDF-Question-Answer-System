#  RAG-based Document Question-Answering System

This project is a **Retrieval-Augmented Generation (RAG)** based QA system that allows users to **upload a PDF document** and **ask questions** about its contents. It leverages **LangChain**, **FAISS**, and **Hugging Face Transformers** to deliver accurate answers based on the uploaded document.

---

##  Features

-  Upload any PDF document.
-  Ask natural language questions based on the document content.
-  Uses semantic search to retrieve relevant text chunks.
-  Answers generated using a pre-trained FLAN-T5 model.
-  Interactive UI built with Gradio.

---

##  Tech Stack

| Component      | Description |
|----------------|-------------|
| **LangChain**  | Orchestrates document loading, text splitting, embedding, and QA chaining. |
| **FAISS**      | Stores document chunks as vector embeddings for fast similarity search. |
| **Sentence Transformers** | Converts document text into vector embeddings. |
| **Hugging Face Transformers** | Provides the FLAN-T5 model for answer generation. |
| **Gradio**     | Builds the user interface for uploading and querying documents. |
| **PyPDF**      | Extracts text content from PDF files. |

---

##  How It Works

1. **Upload PDF**: The system loads and splits the document into manageable chunks.
2. **Embed Chunks**: Each chunk is embedded using a Sentence Transformer model.
3. **Store in FAISS**: Embeddings are stored in a FAISS vector index.
4. **Ask a Question**: A query is matched to the most relevant document chunks.
5. **Answer Generated**: The retrieved content is passed to a FLAN-T5 model for answer generation.

---

## 🧪 Installation

```bash
pip install langchain faiss-cpu sentence-transformers transformers pypdf gradio
