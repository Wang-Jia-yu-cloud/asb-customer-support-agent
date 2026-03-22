import os
import json
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

SYSTEM_PROMPT = """You are Kiri, a friendly senior customer support specialist at ASB Bank New Zealand.

Rules:
- ALWAYS start with a brief natural greeting like "Hi there!" or "Hey!"
- Talk like a real person: warm, direct, clear
- Use contractions: you're, it's, here's, don't
- When giving instructions for multiple methods (e.g. app vs internet banking), use clear headers like "**FastNet Classic:**" and "**ASB Mobile app:**" followed by numbered steps under each
- Never mix the numbering across different methods — each method starts from 1
- Number steps clearly when giving instructions
- Do NOT include any URLs or links
- Do NOT sign off with your name — this is chat, not email
- Never use corporate filler like "Certainly!", "I hope this helps"
- Use ONLY the knowledge provided below as your source
- Do NOT invent specific ASB policies
- If the knowledge base doesn't have the answer, say so honestly and suggest contacting ASB
- If the user's message is a short follow-up (like "mail", "yes", "no"), use the conversation history to understand what they mean"""


@lru_cache(maxsize=512)
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
            "score": round(m.score, 3),
        }
        for m in results.matches
        if m.score > 0.72
    ]


def build_search_query(user_message: str, chat_history: list) -> str:
    if not chat_history or len(user_message.split()) > 5:
        if len(user_message.split()) <= 3:
            return user_message
    
    recent = ""
    if chat_history:
        recent = "\n".join([
            f"{m['role'].upper()}: {str(m['content'])[:200]}"
            for m in chat_history[-4:]
            if m["role"] in ["user", "assistant"]
        ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a conversation history and the latest user message, "
                    "write a single clear search query for an ASB bank knowledge base. "
                    "The query should capture the full intent including context from history. "
                    "Output only the search query, nothing else."
                )
            },
            {
                "role": "user",
                "content": f"History:\n{recent}\n\nLatest message: {user_message}"
            }
        ]
    )
    return response.choices[0].message.content.strip()


def classify(message: str) -> str:
    msg = message.lower()
    if any(x in msg for x in ["human", "person", "real person", "speak to", "talk to", "call me", "call asb"]):
        return "escalate"
    if any(x in msg for x in ["not working", "failed", "error", "missing", "gone", "blocked", "fraud", "scam", "stolen", "can't login", "locked out"]):
        return "complaint"
    return "faq"


def run_crew(user_message: str, chat_history=None, state=None) -> tuple:
    if state is None:
        state = {}
    if chat_history is None:
        chat_history = []

    category = classify(user_message)

    if category == "escalate":
        return (
            "Hi there! No worries, let me get you connected with someone who can help.\n\n"
            f"{ASB_CONTACT}",
            state
        )

    if category == "complaint":
        docs = search(build_search_query(user_message, chat_history))
        self_help = ""
        if docs:
            self_help = (
                "\n\nIn the meantime, here's something that might help:\n"
                f"{docs[0]['answer'][:300]}"
            )
        return (
            f"Hi there! That doesn't sound right — let's get this sorted.{self_help}\n\n"
            f"For urgent help, contact ASB directly:\n{ASB_CONTACT}",
            state
        )

    query = build_search_query(user_message, chat_history)
    docs = search(query)

    if docs:
        context = "\n\n---\n\n".join([
            f"Q: {d['question']}\nA: {d['answer']}"
            for d in docs
        ])
    else:
        context = "No relevant information found in the knowledge base."

    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nKnowledge base results:\n{context}"
        }
    ]

    for msg in chat_history[-6:]:
        if msg["role"] in ["user", "assistant"] and msg.get("content"):
            messages.append({
                "role": msg["role"],
                "content": str(msg["content"])[:500]
            })

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3,
        messages=messages,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip(), state
