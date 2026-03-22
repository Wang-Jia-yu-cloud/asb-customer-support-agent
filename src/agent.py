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
- When listing steps for multiple methods, ALWAYS use this exact format:
  **Method Name:**
  1. First step
  2. Second step

  **Another Method:**
  1. First step
  2. Second step

- When comparing items (like debit vs credit card), use this format:
  **Item Name:**
  - Point one
  - Point two

  **Another Item:**
  - Point one
  - Point two

- NEVER continue numbering across different sections — each section always starts from 1
- NEVER use numbers for category headers — headers are always bold text, numbers are only for steps
- Do NOT include any URLs or links
- Do NOT sign off with your name
- Never use "Certainly!", "I hope this helps"
- Use ONLY the knowledge provided — do NOT invent ASB policies
- If the knowledge base has no answer, say so and suggest contacting ASB
- Focus on PERSONAL banking unless the user specifically mentions business"""

AMBIGUOUS_TOPICS = {
    "card": {
        "question": "Are you looking to apply for a personal credit card, debit card, or a business card?",
        "specifics": ["credit", "debit", "business", "visa", "personal"],
    },
    "loan": {
        "question": "Are you asking about a home loan, personal loan, or business loan?",
        "specifics": ["home", "personal", "business", "mortgage", "property"],
    },
    "statement": {
        "question": "Are you after a transaction statement, credit card statement, or loan statement?",
        "specifics": ["transaction", "credit card", "loan", "account", "bank"],
    },
    "account": {
        "question": "Are you asking about a savings account, transaction account, or term deposit?",
        "specifics": ["savings", "transaction", "term deposit", "cheque", "everyday"],
    },
    "transfer": {
        "question": "Are you making a domestic transfer or an international money transfer?",
        "specifics": ["international", "domestic", "overseas", "nz", "local"],
    },
    "alert": {
        "question": "Would you like email alerts, SMS alerts, or push notifications?",
        "specifics": ["email", "sms", "push", "notification", "text"],
    },
    "limit": {
        "question": "Are you asking about daily payment limits or credit card limits?",
        "specifics": ["daily", "credit card", "payment", "transaction"],
    },
}


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


def needs_clarification(user_message: str, chat_history: list) -> str | None:
    msg = user_message.lower()
    recent_context = " ".join([
        str(m.get("content", "")).lower()
        for m in chat_history[-6:]
        if m["role"] in ["user", "assistant"]
    ])
    combined = msg + " " + recent_context

    for keyword, config in AMBIGUOUS_TOPICS.items():
        if keyword in msg:
            already_specified = any(
                word in combined
                for word in config["specifics"]
            )
            if not already_specified:
                return config["question"]
    return None


def build_search_query(user_message: str, chat_history: list) -> str:
    recent_turns = []
    for m in chat_history[-6:]:
        if m["role"] in ["user", "assistant"] and m.get("content"):
            recent_turns.append(f"{m['role'].upper()}: {str(m['content'])[:200]}")

    if not recent_turns:
        return user_message

    context = "\n".join(recent_turns)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are building a search query for an ASB bank knowledge base. "
                    "Given the conversation history and latest message, "
                    "write ONE clear specific search query that captures the full intent. "
                    "Include specific details like product type if mentioned. "
                    "Output only the search query, nothing else."
                )
            },
            {
                "role": "user",
                "content": f"Conversation:\n{context}\n\nLatest message: {user_message}"
            }
        ]
    )
    return response.choices[0].message.content.strip()


def classify(message: str) -> str:
    msg = message.lower()
    if any(x in msg for x in ["human", "real person", "speak to someone", "talk to someone", "call asb", "phone asb"]):
        return "escalate"
    if any(x in msg for x in ["not working", "failed", "error", "missing", "gone", "blocked", "fraud", "scam", "stolen", "locked out"]):
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
            self_help = f"\n\nHere's something that might help in the meantime:\n{docs[0]['answer'][:300]}"
        return (
            f"Hi there! That doesn't sound right — let's get this sorted.{self_help}\n\n"
            f"For urgent help, contact ASB directly:\n{ASB_CONTACT}",
            state
        )

    clarification = needs_clarification(user_message, chat_history)
    if clarification:
        return f"Hey! {clarification}", state

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
            "content": f"{SYSTEM_PROMPT}\n\nKnowledge:\n{context}"
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
