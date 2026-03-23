# ASB Customer Support Agent

An AI-powered customer support chatbot built with RAG (Retrieval-Augmented Generation), deployed on Streamlit Cloud.

## Live Demo

[asb-customer-support-agent.streamlit.app](https://asb-customer-support-agent.streamlit.app)

## Overview

This project simulates an ASB Bank virtual support specialist named Kiri. It answers customer questions about ASB banking products and services using a knowledge base scraped from ASB's public Help pages.

The system retrieves relevant information from a vector database and generates natural, conversational responses using OpenAI's GPT-4o-mini.

## Architecture
```
User Question
      ↓
Keyword-based Intent Classification
      ↓
Context-aware Query Rewriting (GPT-4o-mini)
      ↓
Vector Search (Pinecone)
      ↓
Response Generation (GPT-4o-mini)
      ↓
Answer
```

## Key Features

- **RAG Pipeline** — 1,039 ASB help articles scraped, embedded, and stored in Pinecone
- **Context-aware search** — query rewriting uses conversation history for follow-up questions
- **Ambiguity handling** — asks clarifying questions when the user's intent is unclear (e.g. "apply for a card" → asks which type)
- **Conversation memory** — passes the last 6 turns to the LLM for multi-turn coherence
- **Intent classification** — routes complaints and escalation requests to appropriate responses
- **ASB-style UI** — built to match ASB's Virtual Support interface with a fixed header and chat bubbles

## Tech Stack

| Component | Tool |
|---|---|
| Web scraping | Python, requests, BeautifulSoup |
| Vector database | Pinecone |
| Embeddings | OpenAI text-embedding-ada-002 |
| LLM | OpenAI GPT-4o-mini |
| Frontend | Streamlit |
| Deployment | Streamlit Cloud |

## Project Structure
```
asb-customer-support-agent/
├── app.py                  # Streamlit UI
├── src/
│   ├── agent.py            # RAG pipeline and response generation
│   ├── scraper.py          # ASB Help page scraper
│   └── rag_builder.py      # Pinecone index builder
├── data/
│   └── asb_faq_full.json   # Scraped knowledge base (1,039 articles)
└── requirements.txt
```

## How to Run Locally

1. Clone the repo
```bash
git clone https://github.com/Wang-Jia-yu-cloud/asb-customer-support-agent.git
cd asb-customer-support-agent
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables — create a `.env` file
```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
OPENAI_MODEL_NAME=gpt-4o-mini
```

4. Run the app
```bash
streamlit run app.py
```

## Data Pipeline

The knowledge base was built in three steps:

1. **Scrape** — collected all article links across 52 ASB Help tags, then scraped 1,039 unique articles
2. **Clean** — removed duplicate titles, social sharing noise, and last-updated timestamps
3. **Embed** — each article was embedded using OpenAI and uploaded to Pinecone in batches of 50

## Author

Levon Wang — [LinkedIn](https://linkedin.com/in/levonwang) · [GitHub](https://github.com/Wang-Jia-yu-cloud)
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("Created: README.md")
