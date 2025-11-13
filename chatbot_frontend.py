import streamlit as st
import requests


st.set_page_config(page_title="SILQFi Chatbot", page_icon="💬", layout="centered")
st.title("SILQFi Customer Support Chatbot")

API_URL = "http://localhost:8000/chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

customer_id = st.text_input("Customer ID", value="CUST001")
if "user_message" not in st.session_state:
    st.session_state.user_message = ""

def send_message():
    user_message = st.session_state.user_message
    if user_message.strip():
        payload = {"customer_id": customer_id, "message": user_message}
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            data = response.json()
            ai_response = data.get("response", "No response from server.")
            query_type = data.get("query_type", "N/A")
            escalation = data.get("escalation_needed", False)
            st.session_state.chat_history.append(("user", user_message))
            st.session_state.chat_history.append(("ai", ai_response))
            st.session_state.chat_history.append(("meta", f"Type: {query_type} | Escalation: {escalation}"))
        except Exception as e:
            st.error(f"Error: {e}")
        st.session_state.user_message = ""

# Main chat area
chat_container = st.container()
with chat_container:
    st.subheader("Conversation History")
    for entry in st.session_state.chat_history:
        if entry[0] == "user":
            st.markdown(f"<div style='text-align:left; background:#e6f7ff; padding:8px; border-radius:8px; margin-bottom:4px;'><b>You:</b> {entry[1]}</div>", unsafe_allow_html=True)
        elif entry[0] == "ai":
            st.markdown(f"<div style='text-align:left; background:#f6ffed; padding:8px; border-radius:8px; margin-bottom:4px;'><b>AI:</b> {entry[1]}</div>", unsafe_allow_html=True)
        elif entry[0] == "meta":
            st.caption(entry[1])
    st.markdown("---")

# Input at the bottom
input_container = st.container()
with input_container:
    st.text_input(
        "Type your message and press Enter",
        st.session_state.user_message,
        key="user_message",
        on_change=send_message,
        placeholder="Enter your query here..."
    )

st.markdown("---")
st.info("Powered by SIQLFi multi-agent workflow. Backend: FastAPI, Frontend: Streamlit.")
