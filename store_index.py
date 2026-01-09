from src.helper import load_pdf_file, text_split, download_embeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os
import time

# 1️⃣ Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# 2️⃣ Load and preprocess documents
print("Loading PDF documents...")
documents = load_pdf_file(data="Data/")

print("Splitting documents into chunks...")
text_chunks = text_split(documents)

print("Loading embedding model (HuggingFace, 384-dim)...")
embeddings = download_embeddings()

# 3️⃣ Pinecone index configuration
INDEX_NAME = "medical-vector"
DIMENSION = 384
METRIC = "cosine"

# 4️⃣ Initialize Pinecone client
print("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

time.sleep(1)

# 5️⃣ Check if index exists, create if not
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"Index '{INDEX_NAME}' not found. Creating index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to be ready...")
    time.sleep(10)
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# 6️⃣ Store embeddings in Pinecone using LangChain wrapper
print("Upserting document embeddings into Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Document ingestion completed successfully.")
