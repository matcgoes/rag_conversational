# RAG Conversational

A LangChain study project. Chat with PDF documents using a conversational RAG pipeline with persistent session history.

## Stack

- **LangChain** — chains, retrievers, history management
- **Groq (LLaMA 3.3 70B)** — LLM
- **HuggingFace** — `all-MiniLM-L6-v2` embeddings
- **Chroma** — vector store
- **Streamlit** — UI

## Setup

```bash
pip install streamlit langchain langchain-groq langchain-huggingface langchain-chroma langchain-community pypdf python-dotenv
```

Add a `.env` file with your `HF_TOKEN`, then run:

```bash
streamlit run app.py
```

Enter your Groq API key in the interface, upload PDFs, and chat.