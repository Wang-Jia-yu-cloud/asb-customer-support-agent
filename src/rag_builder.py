import os
import json
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "asb-support-agent"
DIMENSION = 1536


def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index: {INDEX_NAME}")
    else:
        print(f"Index already exists: {INDEX_NAME}")
    return pc.Index(INDEX_NAME)


def embed_text(text: str) -> list:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    )
    return response.data[0].embedding


def build_knowledge_base(data_path: str = "data/asb_faq_full.json"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    index = get_or_create_index()

    print(f"Uploading {len(data)} articles to Pinecone...")

    batch = []
    for i, item in enumerate(data):
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        vector = embed_text(content)
        batch.append({
            "id": str(i),
            "values": vector,
            "metadata": {
                "question": item["question"],
                "answer": item["answer"],
                "url": item["url"],
            }
        })

        if len(batch) == 50:
            index.upsert(vectors=batch)
            print(f"  Uploaded {i+1}/{len(data)}...")
            batch = []

    if batch:
        index.upsert(vectors=batch)

    print(f"Done. {len(data)} articles uploaded to Pinecone.")


def query_knowledge_base(query: str, top_k: int = 3) -> list:
    index = pc.Index(INDEX_NAME)
    query_vector = embed_text(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [
        {
            "question": match.metadata["question"],
            "answer": match.metadata["answer"],
            "url": match.metadata["url"],
            "score": match.score,
        }
        for match in results.matches
    ]
