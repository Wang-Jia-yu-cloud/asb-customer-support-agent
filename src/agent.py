import os
import re
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("asb-support-agent")

ASB_CONTACT = """
- Phone: 0800 803 804 (personal banking)
- Phone: 0800 272 149 (business banking)
- Visit your nearest ASB branch
- Online: asb.co.nz/contact-us
"""


@lru_cache(maxsize=256)
def get_embedding(text: str) -> tuple:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    )
    return tuple(response.data[0].embedding)


def search(query: str, top_k: int = 5) -> list:
    vector = list(get_embedding(query))
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            "question": m.metadata["question"],
            "answer": m.metadata["answer"],
        }
        for m in results.matches
    ]


def rewrite_query(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Rewrite the user's question into a clear and specific banking search query. Output only the rewritten query, nothing else."
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )
    return response.choices[0].message.content.strip()


def classify(message: str) -> str:
    msg = message.lower()
    if any(x in msg for x in ["human", "person", "call", "speak", "real", "someone"]):
        return "escalate"
    if any(x in msg for x in ["not working", "failed", "error", "missing", "gone", "blocked", "fraud", "scam", "stolen"]):
        return "complaint"
    if any(x in msg for x in ["loan", "credit", "kiwisaver", "mortgage", "insurance"]):
        return "product"
    return "faq"


def run_crew(user_message: str, chat_history=None) -> str:
    improved_query = rewrite_query(user_message)

    docs = search(improved_query, top_k=5)

    if docs:
        best = docs[0]
        context = f"Q: {best['question']}\nA: {best['answer']}"
    else:
        context = "No relevant information found."

    category = classify(user_message)

    if category in ["complaint", "escalate"]:
        return (
            "Hi there! I can see something's not quite right.\n\n"
            "Here's what you can try:\n"
            "1. Check your internet banking or mobile app\n"
            "2. Restart the app and try again\n\n"
            "If the issue continues, contact ASB directly:\n"
            f"{ASB_CONTACT}"
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Kiri, a friendly senior customer support specialist at ASB Bank New Zealand.\n"
                "Rules:\n"
                "- ALWAYS start with a brief natural greeting like 'Hi there!' or 'Hey!'\n"
                "- Talk like a real person: warm, direct, clear\n"
                "- Use contractions: you're, it's, here's, don't\n"
                "- Number steps clearly when giving instructions\n"
                "- Do NOT include any URLs or links\n"
                "- Do NOT sign off with your name — this is chat not email\n"
                "- Never use corporate filler like 'Certainly!', 'I hope this helps'\n"
                "- ONLY use the knowledge provided below — do not make things up\n"
                "- If unsure, say so honestly and suggest contacting ASB directly\n\n"
                f"Knowledge:\n{context}"
            )
        }
    ]

    if chat_history:
        for msg in chat_history[-6:]:
            if msg["role"] in ["user", "assistant"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3,
        messages=messages,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()
