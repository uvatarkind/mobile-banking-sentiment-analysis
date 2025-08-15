# src/app.py
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag_pipeline import rag_answer

import streamlit as st

# ---------------------
# Page setup
# ---------------------
st.set_page_config(page_title="CrediTrust AI Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 CrediTrust Complaint Insights Chatbot")
st.caption("Ask me about customer complaints. I'll provide insights and show you where I found them.")

# ---------------------
# Chat history
# ---------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
with st.sidebar:
    st.markdown("### Options")
    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------
# Chat display
# ---------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ---------------------
# User input
# ---------------------
query = st.chat_input("Ask a question about complaints...")

if query:
    st.chat_message("user").markdown(f"💬 {query}")
    st.session_state.messages.append({"role": "user", "content": f"💬 {query}"})

    with st.spinner("🤔 Analyzing complaints..."):
        result = rag_answer(query)
        answer = result["answer"]
        chunks = result["retrieved_chunks"][:2]

        # Build response markdown with inline sources
        response_md = f"**💡 Answer:**\n\n{answer}\n\n---\n**📄 Sources:**\n"
        for i, chunk in enumerate(chunks, 1):
            response_md += f"\n> **Source {i}:** {chunk.strip()}\n"

    with st.chat_message("assistant"):
        st.markdown(response_md, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response_md})
