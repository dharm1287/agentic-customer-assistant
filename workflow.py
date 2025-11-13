"""
Multi-Agent Customer Support System using LangGraph
Demonstrates: Multi-agent workflow, tool calling, RAG, memory, guardrails
"""

import os
from typing import TypedDict, Annotated, Sequence, List, Dict, Any
from operator import add
import json
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load INTERNAL_DOCS from text file (line by line)
INTERNAL_DOCS = []
internal_docs_path = os.path.join(os.path.dirname(__file__), "internal_docs.txt")
if os.path.exists(internal_docs_path):
    with open(internal_docs_path, "r", encoding="utf-8") as f:
        INTERNAL_DOCS = [line.strip() for line in f if line.strip()]


# Load CUSTOMER_LOGS from JSON file
CUSTOMER_LOGS = {}
customer_logs_path = os.path.join(os.path.dirname(__file__), "customer_logs.json")
if os.path.exists(customer_logs_path):
    with open(customer_logs_path, "r", encoding="utf-8") as f:
        CUSTOMER_LOGS = json.load(f)

# Mock chat history
CHAT_HISTORY = {
    "CUST001": [
        {"role": "user", "content": "How do I upgrade my plan?", "timestamp": "2024-11-09"},
        {"role": "assistant", "content": "You can upgrade in Account Settings under Billing.", "timestamp": "2024-11-09"},
    ],
    "CUST002": [
        {"role": "user", "content": "Is there a mobile app?", "timestamp": "2024-11-07"},
        {"role": "assistant", "content": "We're currently developing a mobile app. Expected release Q1 2025.", "timestamp": "2024-11-07"},
    ],
}

# ============================================================================
# VECTOR DATABASE SETUP (RAG Implementation)
# ============================================================================

class VectorDatabase:
    """Vector database for semantic search using FAISS and OpenAI embeddings."""
    
    def __init__(self, documents: List[str]):
        print("Initializing vector database...")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        
        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Create Document objects with metadata
        docs = []
        for i, doc_text in enumerate(documents):
            chunks = text_splitter.split_text(doc_text)
            for j, chunk in enumerate(chunks):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": f"doc_{i}",
                        "chunk": j,
                        "doc_type": self._classify_doc_type(chunk)
                    }
                ))
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        print(f"Vector database initialized with {len(docs)} document chunks")
    
    def _classify_doc_type(self, text: str) -> str:
        """Classify document type based on content."""
        text_lower = text.lower()
        if any(word in text_lower for word in ["pricing", "plan", "cost", "$"]):
            return "billing"
        elif any(word in text_lower for word in ["technical", "requirement", "system", "software"]):
            return "technical"
        elif any(word in text_lower for word in ["policy", "refund", "cancel"]):
            return "policy"
        elif any(word in text_lower for word in ["api", "integration", "oauth"]):
            return "integration"
        elif any(word in text_lower for word in ["security", "encryption", "compliance"]):
            return "security"
        else:
            return "general"
    
    def similarity_search(self, query: str, k: int = 3, filter_type: str = None) -> List[Document]:
        """Perform similarity search in vector database."""
        if filter_type:
            # Filter by document type
            filter_dict = {"doc_type": filter_type}
            results = self.vectorstore.similarity_search(
                query, 
                k=k, 
                filter=filter_dict
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """Perform similarity search with relevance scores."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
    
    def max_marginal_relevance_search(self, query: str, k: int = 3, fetch_k: int = 10) -> List[Document]:
        """Perform MMR search for diverse results."""
        results = self.vectorstore.max_marginal_relevance_search(
            query, 
            k=k, 
            fetch_k=fetch_k
        )
        return results

# Initialize global vector database
print("Setting up RAG system...")
vector_db = VectorDatabase(INTERNAL_DOCS)
print("RAG system ready!\n")

# ============================================================================
# TOOLS
@tool
def create_support_ticket(customer_id: str, query: str) -> str:
    """Create a support ticket for the customer and log it in customer_logs.json."""
    ticket = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "action": "support_ticket",
        "query": query
    }
    # Update in-memory CUSTOMER_LOGS
    if customer_id in CUSTOMER_LOGS:
        CUSTOMER_LOGS[customer_id].append(ticket)
    else:
        CUSTOMER_LOGS[customer_id] = [ticket]
    # Persist to customer_logs.json
    customer_logs_path = os.path.join(os.path.dirname(__file__), "customer_logs.json")
    try:
        with open(customer_logs_path, "w", encoding="utf-8") as f:
            json.dump(CUSTOMER_LOGS, f, indent=2)
        return f"Support ticket created for {customer_id}."
    except Exception as e:
        return f"Error creating support ticket: {e}"
# ============================================================================

@tool
def search_documentation(query: str, k: int = 3) -> str:
    """Search internal product documentation using semantic similarity.
    
    Args:
        query: The search query
        k: Number of relevant documents to return (default: 3)
    
    Returns:
        Relevant documentation with similarity scores
    """
    try:
        # Perform semantic search with scores
        results = vector_db.similarity_search_with_score(query, k=k)
        
        if not results:
            return "No relevant documentation found for this query."
        
        # Format results with relevance scores
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            # Lower score means more similar (FAISS uses L2 distance)
            similarity_percentage = max(0, 100 - (score * 10))
            formatted_results.append(
                f"[Result {i}] (Relevance: {similarity_percentage:.1f}%)\n"
                f"Type: {doc.metadata.get('doc_type', 'general')}\n"
                f"Content: {doc.page_content}\n"
            )
        
        response = f"Found {len(results)} relevant document(s) using semantic search:\n\n"
        response += "\n".join(formatted_results)
        
        return response
    
    except Exception as e:
        return f"Error searching documentation: {str(e)}"

@tool
def search_documentation_by_category(query: str, category: str) -> str:
    """Search documentation filtered by specific category.
    
    Args:
        query: The search query
        category: Document category (billing, technical, policy, integration, security, general)
    
    Returns:
        Relevant documentation from the specified category
    """
    try:
        valid_categories = ["billing", "technical", "policy", "integration", "security", "general"]
        
        if category.lower() not in valid_categories:
            return f"Invalid category. Valid categories: {', '.join(valid_categories)}"
        
        results = vector_db.similarity_search(query, k=3, filter_type=category.lower())
        
        if not results:
            return f"No relevant {category} documentation found for this query."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(
                f"[{category.upper()} Result {i}]\n"
                f"{doc.page_content}\n"
            )
        
        response = f"Found {len(results)} relevant {category} document(s):\n\n"
        response += "\n".join(formatted_results)
        
        return response
    
    except Exception as e:
        return f"Error searching {category} documentation: {str(e)}"

@tool
def search_documentation_diverse(query: str) -> str:
    """Search documentation using Maximum Marginal Relevance for diverse results.
    
    This is useful when you want varied perspectives on a topic rather than
    similar documents. Good for exploratory queries.
    
    Args:
        query: The search query
    
    Returns:
        Diverse relevant documentation
    """
    try:
        results = vector_db.max_marginal_relevance_search(query, k=3, fetch_k=10)
        
        if not results:
            return "No relevant documentation found for this query."
        
        formatted_results = []
        for i, doc in enumerate(results, 1):
            formatted_results.append(
                f"[Diverse Result {i}]\n"
                f"Type: {doc.metadata.get('doc_type', 'general')}\n"
                f"Content: {doc.page_content}\n"
            )
        
        response = f"Found {len(results)} diverse document(s):\n\n"
        response += "\n".join(formatted_results)
        
        return response
    
    except Exception as e:
        return f"Error searching documentation: {str(e)}"

@tool
def get_customer_logs(customer_id: str) -> str:
    """Retrieve customer activity logs and history."""
    if customer_id in CUSTOMER_LOGS:
        logs = CUSTOMER_LOGS[customer_id]
        return f"Customer {customer_id} logs:\n" + json.dumps(logs, indent=2)
    return f"No logs found for customer {customer_id}."

@tool
def get_chat_history(customer_id: str) -> str:
    """Retrieve previous chat history with the customer."""
    if customer_id in CHAT_HISTORY:
        history = CHAT_HISTORY[customer_id]
        formatted = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        return f"Previous conversations with {customer_id}:\n{formatted}"
    return f"No previous conversations found for customer {customer_id}."

@tool
def check_refund_eligibility(customer_id: str, purchase_date: str) -> str:
    """Check if customer is eligible for a refund based on purchase date."""
    try:
        purchase = datetime.strptime(purchase_date, "%Y-%m-%d")
        today = datetime.now()
        days_since_purchase = (today - purchase).days
        
        if days_since_purchase <= 30:
            return f"Customer {customer_id} IS eligible for a full refund. Purchase was {days_since_purchase} days ago."
        else:
            return f"Customer {customer_id} is NOT eligible for a full refund. Purchase was {days_since_purchase} days ago (>30 days). Partial refund may be available."
    except:
        return "Invalid date format. Please use YYYY-MM-DD format."

@tool
def escalate_to_human(reason: str, customer_id: str) -> str:
    """Escalate the conversation to a human agent."""
    return f"ESCALATION REQUEST: Customer {customer_id} issue escalated. Reason: {reason}. Human agent will take over shortly."

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

def content_safety_check(message: str) -> Dict[str, Any]:
    """Check message content for safety violations."""
    violations = []
    
    # Check for PII exposure
    pii_patterns = ["ssn", "social security", "credit card"]
    if any(pattern in message.lower() for pattern in pii_patterns):
        violations.append("PII_EXPOSURE")
    
    # Check for abusive content
    abusive_words = ["hate", "abuse", "threat"]
    if any(word in message.lower() for word in abusive_words):
        violations.append("ABUSIVE_CONTENT")
    
    # Check for unauthorized requests
    unauthorized = ["delete all users", "access admin", "bypass"]
    if any(phrase in message.lower() for phrase in unauthorized):
        violations.append("UNAUTHORIZED_REQUEST")
    
    return {
        "safe": len(violations) == 0,
        "violations": violations,
        "message": message
    }

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    customer_id: str
    query_type: str
    needs_escalation: bool
    safety_check: Dict[str, Any]
    context: Dict[str, Any]
    next_agent: str

# ============================================================================
# AGENT NODES
# ============================================================================

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Bind tools to LLM
tools = [
    search_documentation,
    search_documentation_by_category,
    search_documentation_diverse,
    get_customer_logs,
    get_chat_history,
    check_refund_eligibility,
    escalate_to_human,
    create_support_ticket
]
llm_with_tools = llm.bind_tools(tools)

def router_agent(state: AgentState) -> AgentState:
    """Routes incoming queries to appropriate specialized agent."""
    messages = state["messages"]
    last_message = messages[-1].content
    
    print(f"\n[DEBUG router_agent] Input messages ({len(messages)} messages):")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}" + (f" (tool_calls: {len(msg.tool_calls)})" if has_tool_calls else ""))
    
    # Safety check
    safety = content_safety_check(last_message)
    state["safety_check"] = safety
    
    if not safety["safe"]:
        state["next_agent"] = "safety_agent"
        return state
    
    # Classify query type
    # Heuristic: If query mentions system requirements, OS, RAM, technical setup, route to TECHNICAL
    technical_keywords = [
        "system requirements", "requirements", "minimum requirements", "run your software", "run software", "os", "windows", "macos", "linux", "ram", "cpu", "memory", "technical", "setup", "install", "installation", "hardware", "specs", "specifications", "performance", "optimization", "internet connection", "compatible", "supported", "platform", "device", "operating system", "configuration", "troubleshoot", "error", "bug", "crash", "not working", "doesn't work", "cannot start", "can't start"
    ]
    matched_keywords = [kw for kw in technical_keywords if kw in last_message.lower()]
    print(f"[DEBUG router_agent] Technical keyword matches: {matched_keywords}")
    if matched_keywords:
        query_type = "TECHNICAL"
    else:
        system_prompt = """You are a query classifier. Classify the customer query into ONE of these categories:
        - TECHNICAL: Technical issues, bugs, errors
        - BILLING: Billing, pricing, refunds, payments
        - GENERAL: General questions, features, documentation
        
        Respond with only the category name."""
        classification_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ]
        response = llm.invoke(classification_messages)
        query_type = response.content.strip().upper()
    state["query_type"] = query_type
    # Route to appropriate agent
    if query_type == "TECHNICAL":
        state["next_agent"] = "technical_agent"
    elif query_type == "BILLING":
        state["next_agent"] = "billing_agent"
    else:
        state["next_agent"] = "general_agent"
    return state

def technical_agent(state: AgentState) -> AgentState:
    """Handles technical support queries."""
    messages = state["messages"]
    customer_id = state.get("customer_id", "UNKNOWN")
    
    system_prompt = f"""You are a technical support specialist. Help the customer with technical issues.
    Customer ID: {customer_id}
    
    Use the available tools to:
    1. Use search_documentation for semantic search of technical docs (automatically finds relevant content)
    2. Use search_documentation_by_category with category='technical' for technical-specific searches
    3. Check customer logs for technical issues
    4. Review chat history for context
    
    The search tools use vector embeddings for semantic similarity, so they understand intent and context.
    You don't need exact keyword matches.
    
    If the issue is complex or requires system access, use escalate_to_human tool.
    Be clear, helpful, and provide step-by-step instructions."""
    
    # Check if this is a follow-up after tool execution (has ToolMessages)
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
    if has_tool_results:
        # Remove all AIMessage with tool_calls and their immediately following ToolMessages
        filtered = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Skip this AIMessage and all immediately following ToolMessages
                i += 1
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    i += 1
                continue
            filtered.append(msg)
            i += 1
        agent_messages = [SystemMessage(content=system_prompt)] + filtered

        # Extract semantic search context from ToolMessages
        context_msgs = [msg for msg in messages if isinstance(msg, ToolMessage) and msg.content]
        context_text = "\n\n".join([msg.content for msg in context_msgs]) if context_msgs else "No relevant context found."

        # If a support ticket was just created, confirm it in the response
        ticket_confirmation = None
        for msg in context_msgs:
            if "Support ticket created for" in msg.content:
                ticket_confirmation = msg.content
                break

        if ticket_confirmation:
            summary_prompt = f"""
Your support ticket has been created for your issue under the identifier {customer_id}.
{ticket_confirmation}
If you need further assistance or updates regarding your account access, please let me know!
"""
            summary_messages = [SystemMessage(content=summary_prompt)] + filtered
            response = llm.invoke(summary_messages)
            response = AIMessage(content=response.content)
        else:
            # Compose a summary response including context
            summary_prompt = f"""
You are a technical support specialist. The following context was retrieved from semantic search and tools:\n\n{context_text}\n\nSummarize and answer the customer's technical question using this context. Be clear and specific. If the context does not answer the question, say so explicitly.
"""
            summary_messages = [SystemMessage(content=summary_prompt)] + filtered
            response = llm.invoke(summary_messages)
            response = AIMessage(content=response.content)
    else:
        agent_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(agent_messages)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "escalate_to_human":
                state["needs_escalation"] = True
    state["messages"] = state["messages"] + [response]
    return state

def billing_agent(state: AgentState) -> AgentState:
    """Handles billing and payment queries."""
    messages = state["messages"]
    customer_id = state.get("customer_id", "UNKNOWN")
    
    system_prompt = f"""You are a billing support specialist. Help the customer with billing, pricing, and refunds.
    Customer ID: {customer_id}
    
    Use the available tools to:
    1. Use search_documentation for semantic search of billing/pricing docs
    2. Use search_documentation_by_category with category='billing' for billing-specific searches
    3. Check customer logs for purchase history
    4. Check refund eligibility when needed
    5. Review chat history for context
    
    The vector search understands natural language queries about pricing, refunds, and billing.
    
    Be empathetic and clear about policies. If refund requested, always check eligibility first.
    For complex billing issues, use escalate_to_human tool."""
    
    # Check if this is a follow-up after tool execution (has ToolMessages)
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
    if has_tool_results:
        # Remove all AIMessage with tool_calls and their immediately following ToolMessages
        filtered = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Skip this AIMessage and all immediately following ToolMessages
                i += 1
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    i += 1
                continue
            filtered.append(msg)
            i += 1
        agent_messages = [SystemMessage(content=system_prompt)] + filtered

        # Extract semantic/tool context from ToolMessages
        context_msgs = [msg for msg in messages if isinstance(msg, ToolMessage) and msg.content]
        context_text = "\n\n".join([msg.content for msg in context_msgs]) if context_msgs else "No relevant context found."

        # Compose a summary response including context
        summary_prompt = f"""
You are a billing support specialist. The following context was retrieved from semantic search and tools:\n\n{context_text}\n\nSummarize and answer the customer's billing question using this context. Be clear and specific. If the context does not answer the question, say so explicitly.
"""
        summary_messages = [SystemMessage(content=summary_prompt)] + filtered
        response = llm.invoke(summary_messages)
        response = AIMessage(content=response.content)
    else:
        agent_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(agent_messages)
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "escalate_to_human":
                state["needs_escalation"] = True
    state["messages"] = state["messages"] + [response]
    return state

def general_agent(state: AgentState) -> AgentState:
    """Handles general queries and information requests."""
    messages = state["messages"]
    customer_id = state.get("customer_id", "UNKNOWN")
    
    # DEBUG: Print message types to diagnose the issue
    # print(f"\n[DEBUG general_agent] Message history ({len(messages)} messages):")
    # for i, msg in enumerate(messages):
    #     msg_type = type(msg).__name__
    #     has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
    #     print(f"  [{i}] {msg_type}" + (f" (tool_calls: {len(msg.tool_calls)})" if has_tool_calls else ""))
    
    system_prompt = f"""You are a general customer support agent. Help the customer with general questions.
    Customer ID: {customer_id}
    
    Use the available tools to:
    1. Use search_documentation for semantic search (finds relevant docs automatically)
    2. Use search_documentation_diverse for broad exploratory queries
    3. Use search_documentation_by_category when you need specific category info
    4. Review chat history for context
    5. Check customer logs if needed
    
    The vector database uses embeddings, so it understands questions semantically, not just keywords.
    
    Be friendly, informative, and concise. Provide accurate information from documentation."""
    
    # Check if this is a follow-up after tool execution (has ToolMessages)
    has_tool_results = any(isinstance(msg, ToolMessage) for msg in messages)
    print(f"[DEBUG general_agent] has_tool_results={has_tool_results}")
    if has_tool_results:
        print(f"[DEBUG general_agent] Using llm (no tools) because we have tool results")
        filtered = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Skip this AIMessage and all immediately following ToolMessages
                i += 1
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    i += 1
                continue
            filtered.append(msg)
            i += 1
        agent_messages = [SystemMessage(content=system_prompt)] + filtered

        # Extract semantic/tool context from ToolMessages
        context_msgs = [msg for msg in messages if isinstance(msg, ToolMessage) and msg.content]
        context_text = "\n\n".join([msg.content for msg in context_msgs]) if context_msgs else "No relevant context found."

        # If a support ticket was just created, confirm it in the response
        ticket_confirmation = None
        for msg in context_msgs:
            if "Support ticket created for" in msg.content:
                ticket_confirmation = msg.content
                break

        if ticket_confirmation:
            summary_prompt = f"""
Your support ticket has been created for your issue under the identifier {customer_id}.
{ticket_confirmation}
If you need further assistance or updates regarding your account access, please let me know!
"""
            summary_messages = [SystemMessage(content=summary_prompt)] + filtered
            response = llm.invoke(summary_messages)
            response = AIMessage(content=response.content)
        else:
            # Compose a summary response including context
            summary_prompt = f"""
You are a general customer support agent. The following context was retrieved from semantic search and tools:\n\n{context_text}\n\nSummarize and answer the customer's general question using this context. Be clear and specific. If the context does not answer the question, say so explicitly.
"""
            summary_messages = [SystemMessage(content=summary_prompt)] + filtered
            response = llm.invoke(summary_messages)
            response = AIMessage(content=response.content)
    else:
        print(f"[DEBUG general_agent] Using llm_with_tools for first pass")
        agent_messages = [SystemMessage(content=system_prompt)] + messages
        response = llm_with_tools.invoke(agent_messages)
    
    state["messages"] = state["messages"] + [response]
    return state

def safety_agent(state: AgentState) -> AgentState:
    """Handles safety violations and inappropriate requests."""
    safety_check = state["safety_check"]
    violations = ", ".join(safety_check["violations"])
    
    response_message = f"""I apologize, but I cannot process this request due to safety concerns: {violations}. 

Please ensure your message:
- Does not contain sensitive personal information (SSN, credit card numbers, passwords)
- Does not include abusive or threatening language
- Does not request unauthorized access to systems

How else may I assist you today?"""
    
    state["messages"] = state["messages"] + [AIMessage(content=response_message)]
    state["needs_escalation"] = True
    return state

def tool_execution_node(state: AgentState) -> AgentState:
    """Execute tools called by agents."""
    messages = state["messages"]
    last_message = messages[-1]
    
    print(f"\n[DEBUG tool_execution_node] Input messages ({len(messages)} messages):")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
        print(f"  [{i}] {msg_type}" + (f" (tool_calls: {len(msg.tool_calls)})" if has_tool_calls else ""))
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_node = ToolNode(tools)
        # Execute tools - ToolNode returns only the tool result messages, not the full history
        result = tool_node.invoke({"messages": messages})
        
        print(f"[DEBUG tool_execution_node] Result messages ({len(result['messages'])} messages):")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls
            print(f"  [{i}] {msg_type}" + (f" (tool_calls: {len(msg.tool_calls)})" if has_tool_calls else ""))
        
        # ToolNode returns only tool result messages, add them all directly
        print(f"[DEBUG tool_execution_node] Adding {len(result['messages'])} tool result messages")
        state["messages"] = state["messages"] + result["messages"]
    
    return state
    
    return state

# ============================================================================
# WORKFLOW GRAPH
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Determine if workflow should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for escalation
    if state.get("needs_escalation", False):
        return "end"
    
    # Check if tools need to be executed
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    return "end"

def create_customer_support_graph():
    """Create the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("technical_agent", technical_agent)
    workflow.add_node("billing_agent", billing_agent)
    workflow.add_node("general_agent", general_agent)
    workflow.add_node("safety_agent", safety_agent)
    workflow.add_node("tools", tool_execution_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        lambda x: x["next_agent"],
        {
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "general_agent": "general_agent",
            "safety_agent": "safety_agent",
        }
    )
    
    # Add edges from specialized agents
    for agent in ["technical_agent", "billing_agent", "general_agent", "safety_agent"]:
        workflow.add_conditional_edges(
            agent,
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )
    
    # Tools back to the agent that called them
    workflow.add_conditional_edges(
        "tools",
        lambda x: x["next_agent"],
        {
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "general_agent": "general_agent",
        }
    )
    
    return workflow.compile()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_support_query(customer_id: str, query: str):
    """Process a customer support query through the agent system."""
    graph = create_customer_support_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "customer_id": customer_id,
        "query_type": "",
        "needs_escalation": False,
        "safety_check": {},
        "context": {},
        "next_agent": ""
    }
    
    print(f"\n{'='*80}")
    print(f"Customer: {customer_id}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    
    result = graph.invoke(initial_state)

    # Find the last AIMessage in the conversation
    final_ai_message = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            final_ai_message = msg.content
            break

    print("\n--- FINAL AI RESPONSE ---\n")
    if final_ai_message:
        print(final_ai_message)
    else:
        print("No AI response found.")

    print(f"\n{'='*80}")
    print(f"Query Type: {result.get('query_type', 'N/A')}")
    print(f"Escalation Needed: {result.get('needs_escalation', False)}")
    print(f"{'='*80}\n")
    return result

if __name__ == "__main__":
    # Example queries demonstrating different agents and RAG capabilities
    
    print("\n" + "="*80)
    print("DEMONSTRATING VECTOR DATABASE RAG SYSTEM")
    print("="*80 + "\n")
    
    # 1. Technical query - demonstrates semantic search
    # print("\n### EXAMPLE 1: Technical Query (Semantic Search) ###")
    # response = run_support_query("CUST001", "What do I need to run your software on my computer?")
    # print("Response:\n", response)
    # print(type(response))
    # # 2. Billing query - demonstrates category-specific search
    # print("\n### EXAMPLE 2: Billing Query ###")
    # run_support_query("CUST001", "I want to request a refund for my purchase on November 5th.")
    
    # 3. General query - demonstrates diverse search
    # print("\n### EXAMPLE 3: General Query (Natural Language) ###")
    # res = run_support_query("CUST002", "How much does the premium subscription cost and what do I get?")
    # print("Response:\n", res)
    
    # # 4. Security query - demonstrates semantic understanding
    # print("\n### EXAMPLE 4: Security Query (Semantic Understanding) ###")
    # run_support_query("CUST001", "Is my data safe with you? What encryption do you use?")
    
    # # 5. Integration query - demonstrates multi-category search
    # print("\n### EXAMPLE 5: Integration Query ###")
    # run_support_query("CUST002", "Can I connect your service to other apps? How many API calls can I make?")
    
    # # 6. Account management query
    print("\n### EXAMPLE 6: Account Management Query ###")
    # run_support_query("CUST001", "I forgot my password, how can I get back into my account?")
    run_support_query("CUST001", "can you fetch the activity logs for me?")

    # print("\n" + "="*80)
    # print("RAG DEMONSTRATION COMPLETE")
    # print("Notice how the vector database finds relevant docs even without exact keyword matches!")
    # print("="*80 + "\n")