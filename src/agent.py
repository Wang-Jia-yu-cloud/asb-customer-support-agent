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

SYSTEM_PROMPT = """You are Kiri, a friendly senior virtual support specialist at ASB Bank New Zealand.

Rules:
- Talk like a real person: warm, direct, clear
- ALWAYS start with a brief natural greeting like "Hi there!" or "Hey!"
- Use contractions: you're, it's, here's, don't
- Number steps clearly when giving instructions
- Do NOT include any URLs or links
- Do NOT sign off with your name — this is chat not email
- Do NOT use corporate filler like "Certainly!", "I hope this helps", "Don't hesitate to reach out"
- End naturally when the answer is complete
- If you don't know, say so honestly and give ASB contact details"""


def classify_query(message: str) -> str:
    msg = message.lower()
    escalate_keywords = ["speak to", "talk to", "real person", "human", "call me", "agent"]
    complaint_keywords = ["failed", "missing", "gone", "blocked", "fraud", "scam", "wrong", "error", "stolen"]

    for kw in escalate_keywords:
        if kw in msg:
            return "escalate"
    for kw in complaint_keywords:
        if kw in msg:
            return "complaint"
    return "faq"


@lru_cache(maxsize=256)
def get_embedding(text: str) -> tuple:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return tuple(response.data[0].embedding)


def search_knowledge_base(query: str, top_k: int = 5) -> str:
    vector = list(get_embedding(query))
    results = index.query(vector=vector, top_k=top_k, include_metadata=True)
    if not results.matches:
        return "No relevant information found."
    output = ""
    for m in results.matches:
        output += f"Q: {m.metadata['question']}\nA: {m.metadata['answer']}\n\n"
    return output.strip()


def run_crew(user_message: str) -> str:
    category = classify_query(user_message)

    if category == "escalate":
        context = ""
        user_prompt = (
            f"The customer wants to speak to a real person.\n\n"
            f"Acknowledge this warmly and give them these contact options:\n{ASB_CONTACT}"
        )
    elif category == "complaint":
        context = search_knowledge_base(user_message)
        user_prompt = (
            f"Customer concern: {user_message}\n\n"
            f"Relevant information from knowledge base:\n{context}\n\n"
            f"Acknowledge their concern, give any relevant self-help steps, "
            f"then provide ASB contact details:\n{ASB_CONTACT}"
        )
    else:
        context = search_knowledge_base(user_message)
        user_prompt = (
            f"Customer question: {user_message}\n\n"
            f"Relevant information from knowledge base:\n{context}\n\n"
            f"Answer the question clearly and completely based on the information above."
        )

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()
