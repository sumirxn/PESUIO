from dotenv import load_dotenv
load_dotenv()
import os
import requests
import json
from pinecone import Pinecone
import cohere


#parsing the pdf file
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    result_type='markdown',
)

file_extractor = {".pdf":parser}
output_docs=SimpleDirectoryReader(input_files=['./data/git-cheat-sheet.pdf'], file_extractor=file_extractor)
docs = output_docs.load_data()
md_text = ""
for doc in docs:
    md_text += doc.text

with open('output.md', 'w') as file_handle:
    file_handle.write(md_text)

print("Markdown file created successfully")

#chunking the parsed markdown
chunk_size = 1000
overlap = 200
def fixed_size_chunking(text, chunk_size, overlap):
    chunks=[]
    start=0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start+=chunk_size-overlap
    return chunks

chunks = fixed_size_chunking(md_text, chunk_size, overlap)
print(f"Number of chunks: {len(chunks)}")

#Embedding
co = cohere.ClientV2(os.getenv('COHERE_API_KEY'))
headers = {
    "Authorization" : f"Bearer {co}", 
    "Content-Type"  :  "application/json"
}
embedded_chunks = []
for chunk in chunks:
    try:
        response = co.embed(
            texts=[chunk],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"]
        )
        embedded_chunks.append(response.embeddings.float_[0])
    except Exception as e:
        print(f"Error embedding chunk: {str(e)}")

print(f"Number of embedded chunks: {len(embedded_chunks)}")

# Save embedded chunks to a JSON file
output_file = 'embedded_chunks.json'

# Prepare data structure for JSON
data_to_save = {
    'chunks': chunks,
    'embeddings': embedded_chunks
}

with open(output_file, 'w') as f:
    json.dump(data_to_save, f)

print(f"Embedded chunks saved to {output_file}")


pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')) 
index_name = "pesuio"  

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
    )

index = pc.Index(index_name)

vectors_to_upsert = [
    {
        'id': f'chunk_{i}',
        'values': embedding,
        'metadata': {'text': chunk}
    }
    for i, (chunk, embedding) in enumerate(zip(chunks, embedded_chunks))
]

index.upsert(vectors=vectors_to_upsert)

print(f"Uploaded {len(vectors_to_upsert)} vectors to Pinecone")