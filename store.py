from src.helper import load_pdf, chunk_splitter, load_embedding_model
from pinecone import Pinecone
import os



PINECONE_API_KEY ='pcsk_2U7yy5_HDTJQWg9WaRgBXi6crVgEk8tzF8oLtEhUeKeAr1rToQNfJk6kxEtiu4FrNnfxxx'

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("test5")

# Load and process data
data = load_pdf("data/")
chunks = chunk_splitter(data)
embeddings = load_embedding_model()

def insert_vectors(chunks, batch_size=50):
    vectors = []

    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_query(chunk.page_content)  # 🔹 Generate embeddings for each chunk
        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {"text": chunk.page_content}  # Metadata for retrieval
        })

        # 🔹 Insert in Batches (Every `batch_size` chunks)
        if len(vectors) >= batch_size:
            index.upsert(vectors=vectors)  # Upsert batch
            print(f"Inserted {len(vectors)} vectors into Pinecone.")
            vectors = []  # Clear batch

    # 🔹 Insert any remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
        print(f"Inserted {len(vectors)} remaining vectors into Pinecone.")

# Insert embeddings into Pinecone
insert_vectors(chunks, batch_size=50)# 🔹 Set batch size to 50