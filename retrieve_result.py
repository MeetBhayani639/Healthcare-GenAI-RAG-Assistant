from groq import Groq
from langchain_pinecone import PineconeVectorStore


# 1️⃣ Retrieve relevant documents from Pinecone
def retrieval_result(query: str, docsearch: PineconeVectorStore, k: int = 3):
    """
    Perform semantic search on Pinecone vector store.
    """
    docs = docsearch.similarity_search(query, k=k)
    return docs


# 2️⃣ Generate final answer using Groq LLM
def result_after_retrieval(api_key: str, query: str, docs):
    """
    Use Groq LLM to generate a grounded answer using retrieved documents.
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    client = Groq(api_key=api_key)

    # Combine retrieved document content
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a healthcare AI assistant.

Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information, say:
"I'm sorry, I don't have enough information to answer that."

User Question:
{query}

Context:
{context}

Answer:
"""

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024,
            top_p=0.8,
            stream=True
        )

        # Collect streamed output
        response_chunks = []

        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                response_chunks.append(chunk.choices[0].delta.content)

        final_response = "".join(response_chunks)
        return final_response

    except Exception as e:
        raise RuntimeError(f"Groq API call failed: {str(e)}")
