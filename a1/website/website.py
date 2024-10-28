from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.groq import Groq
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

Settings.embed_model = CohereEmbedding(
    api_key=os.getenv("COHERE_API_KEY"),
    model_name = "embed-english-v3.0", 
    input_type = "search_query",
)

Settings.llm = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    model = "gemma-7b-it",
    temperature = 0.7
)

def create_rag_system(data_dir="./data"):
    client = QdrantClient (path = "./qdrant_data")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="my_documents",
        dimension=1024  
    )


    documents = SimpleDirectoryReader(data_dir).load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store
    )

    query_engine = index.as_query_engine()
    
    return query_engine

def query_rag(query_engine, question: str):
    # Query the system
    response = query_engine.query(question)
    return response

def main():
    # Initialize the RAG system
    query_engine = create_rag_system()
    
    # Example usage
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        response = query_rag(query_engine, question)
        print(f"\nAnswer: {response}")

if __name__ == "__main__":
    main()
