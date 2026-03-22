import os
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
    filtered = []
    for m in results.matches:
        if m.score > 0.75:
            filtered.append({
                "question": m.metadata["question"],
                "answer": m.metadata["answer"],
                "score": m.score,
            })
    return filtered


def rewrite_query(user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Rewrite the user's question into a clear and specific banking search query. Output only the rewritten query, nothing else."
            },
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()


def update_state(user_message: str, state: dict) -> dict:
    msg = user_message.lower()
    if "card" in msg:
        state["topic"] = "card"
    elif "limit" in msg or "payment" in msg:
        state["topic"] = "payment"
    elif "loan" in msg or "mortgage" in msg:
        state["topic"] = "loan"
    elif "password" in msg or "login" in msg:
        state["topic"] = "access"
    if "personal" in msg:
        state["card_type"] = "personal"
    elif "business" in msg:
        state["card_type"] = "business"
    elif "credit" in msg:
        state["card_type"] = "credit"
    elif "debit" in msg:
        state["card_type"] = "debit"
    return state


def classify(message: str) -> str:
    msg = message.lower()
    if any(x in msg for x in ["human", "person", "call", "speak", "real", "someone", "agent"]):
        return "escalate"
    if any(x in msg for x in ["not working", "failed", "error", "missing", "gone", "blocked", "fraud", "scam", "stolen", "problem", "can't", "unable"]):
        return "complaint"
    return "faq"


def run_crew(user_message: str, chat_history=None, state=None) -> tuple:
    if state is None:
        state = {}

    state = update_state(user_message, state)

    if state.get("topic") == "card" and "card_type" not in state:
        return "Hey! Just to make sure I give you the right info — are you looking at a personal card, business card, credit card, or debit card?", state

    category = classify(user_message)

    if category in ["complaint", "escalate"]:
        reply = (
            "Hi there! Let me get you connected with the right help.\n\n"
            "You can reach ASB directly:\n"
            f"{ASB_CONTACT}"
        )
        return reply, state

    query = rewrite_query(user_message)
    docs = search(query, top_k=5)

    if docs:
        best = docs[0]
        context = f"Q: {best['question']}\nA: {best['answer']}"
    else:
        context = "No relevant information found in the knowledge base."

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
                "- If the knowledge base doesn't have the answer, say so honestly\n"
                "- Ask for clarification if the question is still unclear\n\n"
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

    return response.choices[0].message.content.strip(), state
