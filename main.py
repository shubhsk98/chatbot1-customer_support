# main.py
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Import DB functions
from app import (
    get_user, fetch_user_data, get_kb_docs,
    save_chat_history, insert_user_log, create_user, get_chat_history,db,insert_user_login
)

# LLM + LangChain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
groq_api_key = st.secrets["GROQ_API_KEY"]
# ------------------------------
# Setup
# ------------------------------
load_dotenv()
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant"
)

st.title("🤖 Agentic AI Customer Support Bot")

# ------------------------------
# Hybrid Login System
# ------------------------------
# Login
# Login
if "user" not in st.session_state:
    st.session_state.user = None

if not st.session_state.user:
    identifier = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Check if user exists
        existing_user = db.users.find_one(
            {"$or": [{"username": identifier}, {"email": identifier}]},
            {"_id": 0}
        )

        if existing_user:
            # User exists → check password
            if existing_user["password"] == password:
                st.session_state.user = existing_user
                st.success(f"✅ Logged in as {existing_user.get('username') or existing_user.get('email')}")

                # Store successful login
                insert_user_login(
                    user_id=existing_user["user_id"],
                    login_time=datetime.now().isoformat(),
                    location="Unknown",
                    email=existing_user.get("email"),
                    status="success"
                )
            else:
                st.error("❌ Your password is incorrect. Try again.")

                # Store failed login attempt
                insert_user_login(
                    user_id=existing_user["user_id"],
                    login_time=datetime.now().isoformat(),
                    location="Unknown",
                    email=existing_user.get("email"),
                    status="failed",
                    reason="incorrect password"
                )
        else:
            # Username/email not found
            st.error("❌ Username/Email not found. Please check your credentials.")

            # Store failed login attempt with dummy user_id
            insert_user_login(
                user_id="unknown",
                login_time=datetime.now().isoformat(),
                location="Unknown",
                email=identifier if "@" in identifier else None,
                status="failed",
                reason="username/email not found"
            )

    st.stop()

# Access user_id
user_id = st.session_state.user["user_id"]










# ------------------------------
# KB Embeddings + Retriever
# ------------------------------
if "kb_vectorstore" not in st.session_state:
    kb_docs = get_kb_docs()
    kb_texts = [f"Q: {doc['question']}\nA: {doc['answer']}" for doc in kb_docs]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.kb_vectorstore = FAISS.from_texts(kb_texts, embeddings)

retriever = st.session_state.kb_vectorstore.as_retriever()

# ------------------------------
# Prompts
# ------------------------------
retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Rewrite the user’s question into a standalone query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a friendly and helpful AI support assistant.\n"
     "If the question is about general support, use KB context.\n"
     "If the question is about user-specific details, fetch from DB.\n"
     "If the user wants to log an issue or create a ticket, suggest creating it in DB.\n"
     "Always respond in a polite, clear, and customer-friendly way.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# ------------------------------
# Chains
# ------------------------------
history_aware_retriever = create_history_aware_retriever(llm, retriever, retriever_prompt)
doc_chain = create_stuff_documents_chain(llm, qa_prompt)
retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

conversational_chain = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

# ------------------------------
# Chat Flow
# ------------------------------
#st.subheader("💬 Chat with Support Bot")
user_query = st.text_input("Type your question:")

if user_query:
    # 🔑 1. Fetch user-specific data
    user_data = fetch_user_data(user_id)
    db_context = (
        f"User Logins: {user_data['logins']}\n"
        f"User Purchases: {user_data['purchases']}\n"
        f"User Logs: {user_data['logs']}\n"
    )

    # 🔑 2. Merge DB context with KB retriever
    result = conversational_chain.invoke(
        {"input": user_query, "context": db_context},
        config={"configurable": {"session_id": user_id}}
    )
    final_answer = result["answer"]

    # 🔑 3. Agent action (insert ticket if requested)
    if "create ticket" in user_query.lower() or "log issue" in user_query.lower():
        issue_text = user_query
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        insert_user_log(
            user_id=user_id,
            issue=issue_text,
            resolution="Pending",
            created_at=datetime.now().isoformat()
        )
        final_answer += f"\n\n📝 A support ticket has been created for you (ID: {ticket_id})."

    # 🔑 4. Save chat history in DB
    save_chat_history(user_id, user_query, final_answer)

    # 🔑 5. Display conversation with bubbles
    #st.markdown("## 💬 Conversation History")

    for msg in reversed(st.session_state.chat_history.messages):
        if msg.type == "human":
            st.markdown( 
                f"<div style='background-color:#DCF8C6; padding:10px; border-radius:10px; margin:5px; text-align:right;'>"
                f"🧑 <b>You:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background-color:#E6E6E6; padding:10px; border-radius:10px; margin:5px; text-align:left;'>"
                f"🤖 <b>Bot:</b> {msg.content}</div>",
                unsafe_allow_html=True
            )
    

# ------------------------------
# Show Past Chat History
# ------------------------------
#st.subheader("📜 Your Past Conversations")
past_chats = get_chat_history(user_id)
if past_chats:
    for chat in past_chats:
        st.markdown(f"🧑 **You:** {chat['query']}")
        st.markdown(f"🤖 **Bot:** {chat['response']}")
        st.markdown("---")
else:
    st.info("No previous chats found. Start your first conversation 👇")
