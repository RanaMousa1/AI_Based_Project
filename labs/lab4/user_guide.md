# User Guide: Lab 4 (Minimalist - Groq Version)

## 1. Setup

Install the necessary libraries for Groq, LangChain, and Environment management:
```bash
pip install langchain-groq langchain-community faiss-cpu sentence-transformers wikipedia arxiv python-dotenv
```

## 2. API Keys

I have already created a `.env` file for you with your Groq and LangChain keys. The code will load these automatically using `python-dotenv`.

## 3. Usage

- **App 1 (Support Chatbot)**: `python app1_customer_support.py`
  - Uses RAG to answer from the `./knowledge` folder.
- **App 2 (Research Assistant)**: `python app2_research_assistant.py`
  - Generates reports using Wiki and ArXiv data.

*Note: Models used are Llama3-8b via Groq.*
