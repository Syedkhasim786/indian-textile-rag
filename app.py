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

st.title("🇮🇳 Indian Textile AI Assistant")

query = st.text_input("Ask your question:")

if query:
    results = search(query)
    context = " ".join(results)
    st.write("🧵 Answer:", context)
