import os
import cohere
import requests
from pinecone import Pinecone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = "pesuio"
index = pc.Index(index_name)

co = cohere.ClientV2(os.getenv('COHERE_API_KEY'))
headers = {
    "Authorization" : f'Bearer{co}', 
    "Content-Type" : 'application/json'

}

groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
def retrieve_nearest_chunks(query, top_k=5):
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_embedding = response.embeddings.float_[0]
    
    
    # Query Pinecone for nearest vectors
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return query_response['matches']

def generate_response(query, context):
    prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question. If the answer is not in the context, say "I don't have enough information to answer that question."

Context:
{context}

User's question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-70b-versatile",
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    query = "What is the command to stage my changes?"
    nearest_chunks = retrieve_nearest_chunks(query)

    context = "\n".join([chunk['metadata']['text'] for chunk in nearest_chunks])
    
    rag_response = generate_response(query, context)

    print(f"\nQuery: {query}")
    print("\nRAG Response:")
    print(rag_response)

    print("\nRetrieved chunks:")
    for i, chunk in enumerate(nearest_chunks, 1):
        print(f"\n{i}. Score: {chunk['score']:.4f}")
        print(f"Text: {chunk['metadata']['text'][:200]}...")