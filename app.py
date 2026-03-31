import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/textiles.txt", "r") as f:
    texts = f.readlines()

embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def search(query):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=3)
    return [texts[i] for i in indices[0]]

st.title("Textile AI Assistant")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ChatGPT-style input
query = st.chat_input("Ask anything about textiles...")

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Get response
    results = search(query)
    context = " ".join(results)

    # Show bot response
    st.session_state.messages.append({"role": "assistant", "content": context})
    with st.chat_message("assistant"):
        st.write(context)
