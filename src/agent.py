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
    return filtered[:3]


def detect_intent(user_message: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are an intent classifier for an ASB bank support chatbot.
Return JSON ONLY with this format:
{
  "topic": "brief description of what the user wants",
  "needs_clarification": true or false
}
Rules:
- needs_clarification = true only if the request is genuinely ambiguous and more info is needed
- needs_clarification = false if the intent is clear enough to search for an answer
- topic should be a short, specific description like "reset password", "get account statement", "apply for credit card"
"""
            },
            {"role": "user", "content": user_message}
        ]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"topic": user_message, "needs_clarification": False}


def rewrite_query(user_message: str) -> str:
    if len(user_message.split()) <= 3:
        return user_message
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Rewrite the user's question into a clear and specific banking search query. Output only the query, nothing else."
            },
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()


def classify(message: str) -> str:
    msg = message.lower()
    if any(x in msg for x in ["human", "person", "call", "speak", "real", "someone", "agent"]):
        return "escalate"
    if any(x in msg for x in ["error", "not working", "failed", "problem", "can't", "unable", "missing", "gone", "blocked", "fraud", "stolen"]):
        return "complaint"
    return "faq"


def run_crew(user_message: str, chat_history=None, state=None) -> tuple:
    if state is None:
        state = {}

    try:
        intent = detect_intent(user_message)
        state["topic"] = intent.get("topic", "")
    except Exception:
        intent = {"needs_clarification": False}

    if intent.get("needs_clarification"):
        return "Could you tell me a bit more? I want to make sure I give you the right answer.", state

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
        all_context = "\n\n".join([
            f"Q: {d['question']}\nA: {d['answer']}"
            for d in docs
        ])
        context = all_context
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
                "- Use the knowledge below as your primary source\n"
                "- You may use general banking knowledge if needed\n"
                "- Do NOT invent specific ASB policies\n"
                "- If unsure, say so honestly and suggest contacting ASB\n"
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
                    "content": str(msg["content"])
                })

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        temperature=0.3,
        messages=messages,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip(), state
