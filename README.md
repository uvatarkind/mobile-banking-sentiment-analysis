# 🧠 Intelligent Complaint Analysis for Financial Services (RAG Chatbot)

This project is part of the **10 Academy KAIM 5 - Week 6** challenge. It aims to build a **RAG-powered internal chatbot** that allows teams at **CrediTrust Financial** to analyze thousands of customer complaints efficiently using **LLMs + semantic search**.

---

## 🧠 Business Context

**CrediTrust** receives thousands of monthly complaints about its financial products (credit cards, loans, BNPL, etc.). Product managers and compliance teams currently read complaints manually — costing time and losing insights.

We built a **Retrieval-Augmented Generation (RAG)** system to:
- Let users ask questions in natural language
- Retrieve relevant complaint narratives from a vector DB
- Generate grounded, actionable answers using a language model

---

## 🚀 Final Deliverables

- ✅ A cleaned & filtered CFPB complaints dataset
- ✅ Text embeddings chunked and stored in FAISS vector DB
- ✅ RAG core logic with evaluation
- ✅ Streamlit UI for interactive chat
- ✅ 📄 Final blog-style report

---

## 🧰 Tech Stack

| Component      | Tool/Library                        |
|----------------|-------------------------------------|
| Language Model | `sentence-transformers` (MiniLM)    |
| Vector DB      | `FAISS`                             |
| LLM Framework  | `LangChain`, `Transformers`, `Gemini`|
| UI             | `Streamlit`                         |
| Data           | CFPB Complaints Dataset             |
| Hardware       | GPU-enabled via PyTorch             |

---

## 📁 Project Structure

```bash
.
├── app/
│   ├── __init__.py
│   └── app.py                      # Streamlit app (Task 4)
│
├── src/
│   ├── README.md
│   └── rag_pipeline.py             # RAG core logic (Task 3)
│
├── data/
│   ├── complaints.csv
│   └── filtered_complaints.csv     # Cleaned dataset
│
├── vector_store/
│   ├── faiss_index.bin             # FAISS vector index
│   └── metadata.pkl                # Metadata for complaint chunks
│
├── notebooks/
│   ├── 0.1-eda.ipynb
│   ├── 2.0-embedding.ipynb
│   ├── 3.0-rag-agent.ipynb
│   ├── README.md
│   └── task_1_eda_and_preprocessing.ipynb
│
├── scripts/
│   ├── __init__.py
│   └── README.md
│
├── tests/
│   └── __init__.py
│
├── requirements.txt
├── .env                           # Contains GEMINI_API_KEY
└── README.md
```