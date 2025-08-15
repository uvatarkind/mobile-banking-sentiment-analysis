import os
import pickle
import faiss
import torch
import numpy as np
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# -------------------------
# Load env vars
# -------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# -------------------------
# Load FAISS + metadata
# -------------------------
index = faiss.read_index("vector_store/faiss_index.bin")
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# -------------------------
# Load embedding model
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# -------------------------
# Load Gemini Pro Model
# -------------------------
gemini = genai.GenerativeModel("gemini-2.5-flash")

# -------------------------
# Prompt builder
# -------------------------
def build_prompt(context, question):
    return f"""
You are a helpful assistant for CrediTrust Financial.

Use the following customer complaint excerpts to answer the question.
If the context is not enough, say you don’t have enough information.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}
Answer:
"""

# -------------------------
# Full RAG function
# -------------------------
def rag_answer(question, top_k=5):
    query_vec = embed_model.encode([question])
    distances, indices = index.search(np.array(query_vec).astype("float32"), top_k)

    selected_chunks = [metadata[i]['text'] for i in indices[0]]
    context = "\n\n".join(selected_chunks)
    prompt = build_prompt(context, question)

    response = gemini.generate_content(prompt)
    return {
        "question": question,
        "answer": response.text.strip(),
        "retrieved_chunks": selected_chunks
    }