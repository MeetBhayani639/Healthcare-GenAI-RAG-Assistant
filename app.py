from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from retrieve_result import retrieval_result, result_after_retrieval

# -------------------------
# App & Environment Setup
# -------------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set")

# -------------------------
# Load embeddings & vector store
# -------------------------
embeddings = download_embeddings()

INDEX_NAME = "medical-vector"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_query = request.form["msg"].lower().strip()
    print("User query:", user_query)

    # Greeting handling
    if user_query in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you today?"
    if user_query in ["bye", "goodbye"]:
        return "Goodbye! Take care."
    if "thank" in user_query:
        return "You're welcome! Let me know if you have any other questions."

    # -------- RAG Pipeline --------
    docs = retrieval_result(user_query, docsearch)

    if not docs:
        return "I'm sorry, I couldn't find relevant information for that."

    response = result_after_retrieval(
        api_key=GROQ_API_KEY,
        query=user_query,
        docs=docs
    )

    return response


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
