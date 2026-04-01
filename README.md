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

## Architecture

| Object | Role | Description |
|---|---|---|
| `retriever` | Vector store interface | Wraps Chroma to expose a standardized search interface used by the rest of the pipeline |
| `history_aware_retriever` | Context-aware retrieval | Reformulates the user question into a standalone query using chat history before searching |
| `question_answer_chain` | Answer generation | Stuffs retrieved chunks into the prompt context and generates a grounded response via the LLM |
| `rag_chain` | Full RAG pipeline | Connects the retriever and the answer chain into a single end-to-end RAG object |
| `conversation_rag_chain` | Stateful conversation | Wraps the RAG chain with per-session message history persistence |