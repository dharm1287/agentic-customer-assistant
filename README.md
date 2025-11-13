# Multi-Agent AI Customer Support System

SILQFi bot is a production-grade, multi-agent customer support system built with LangGraph, LangChain, and OpenAI. It demonstrates advanced orchestration of specialized AI agents for technical, billing, general, and safety queries, with tool calling, RAG (Retrieval-Augmented Generation), and robust safety guardrails.

## Features
- **Multi-Agent Workflow:** Router agent classifies queries and routes to specialized agents.
- **Tool Integration:** Agents use tools for documentation search, customer logs, chat history, refund checks, ticket creation, and escalation.
- **RAG (Retrieval-Augmented Generation):** Semantic search over internal documentation using FAISS vector store and OpenAI embeddings.
- **Safety & Moderation:** All queries pass through a safety check for PII, abuse, and unauthorized requests.
- **Support Ticketing:** Agents can create support tickets, which are logged and persisted.
- **Streamlit Frontend:** Modern chat-like UI for customer interaction.
- **FastAPI Backend:** API endpoint for chat queries.

## Project Structure
```
silqfi-bot/
├── chatbot_api.py           # FastAPI backend
├── chatbot_frontend.py      # Streamlit frontend
├── workflow.py              # Main multi-agent workflow
├── requirements.txt         # Python dependencies
├── internal_docs.txt        # Product documentation (for RAG)
├── customer_logs.json       # Customer activity logs
└── README.md                # Project documentation
```

## Quickstart
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the backend:**
   ```bash
   uvicorn chatbot_api:app --reload
   ```
3. **Start the frontend:**
   ```bash
   streamlit run chatbot_frontend.py
   ```
4. **Demo:**
   - Enter a customer ID and query in the Streamlit UI.
   - Example queries:
     - "My app keeps crashing. What are the system requirements?"
     - "I want to request a refund for my purchase on November 5th."
     - "Please create a support ticket I am not able to access my account."

## How It Works
- **Routing:** The router agent classifies the query and routes to the correct agent.
- **Agents:** Each agent uses domain-specific tools and RAG to answer queries.
- **Tools:** Functions for documentation search, logs, chat history, refund checks, ticket creation, and escalation.
- **Safety:** All queries are checked for violations before routing.
- **State:** Message history and context are tracked per query.

## Customization
- **Add new tools:** Define a new `@tool` function in `workflow.py` and add to the tools list.
- **Add new agents:** Create a new agent function and update the workflow graph.
- **Modify safety rules:** Edit the `content_safety_check()` function in `workflow.py`.
- **Change data sources:** Update `internal_docs.txt` and `customer_logs.json` as needed.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

## License
MIT License

## Credits
- LangGraph, LangChain, OpenAI, Streamlit, FastAPI

---
For more details, see the code and comments in `workflow.py`.
