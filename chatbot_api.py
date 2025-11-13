from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
from workflow import run_support_query

app = FastAPI()

class ChatRequest(BaseModel):
    customer_id: str
    message: str
    session_id: Optional[str] = None  # For future session management

class ChatResponse(BaseModel):
    response: str
    query_type: Optional[str] = None
    escalation_needed: Optional[bool] = None

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Call the agent workflow
    result = run_support_query(request.customer_id, request.message)
    # Find the last AIMessage in the conversation
    final_ai_message = None
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            final_ai_message = msg.content
            break
    return ChatResponse(
        response=final_ai_message or "No AI response found.",
        query_type=result.get("query_type"),
        escalation_needed=result.get("needs_escalation")
    )

@app.get("/")
async def root():
    return {"message": "SILQFi Chatbot API is running."}

# To run: uvicorn chatbot_api:app --reload
