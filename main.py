# main.py
import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime
import uuid

# Import DB functions from aap
from app import (
    get_user, fetch_user_data, get_kb_docs,
    save_chat_history, insert_user_log, create_user,get_chat_history,db,insert_user_login,users_collection
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

# ------------------------------
# Setup
# ------------------------------
load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

st.title("Customer Support Agentic AI Bot 🤖 ")


# ==============================
# Login
# ==============================
if "user_id" not in st.session_state:
    st.session_state.user_id = None

if not st.session_state.user_id:
    username = st.text_input("Username or Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(username, password)

        if user:
            # ✅ Existing user login
            st.session_state.user_id = user["user_id"]

            # Update login count
            users_collection.update_one(
                {"user_id": user["user_id"]},
                {"$inc": {"logins": 1}}
            )
            st.success(f"✅ Welcome back, {username}!")

        else:
            # ✅ Auto-register new user
            from uuid import uuid4
            user_id = str(uuid4())
            create_user(username, password)  # store in DB
            st.session_state.user_id = user_id
            st.success(f"🎉 Thank you {username} for creating a New account here! You are now logged in.")

    st.stop()

user_id = st.session_state.user_id


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
        final_answer += f"\n\n A support ticket has been created for you (ID: {ticket_id})."

    # 🔑 4. Save chat history in DB
    save_chat_history(user_id, user_query, final_answer)

    # Display conversation in bubbles
    for msg in st.session_state.chat_history.messages:
        if msg.type == "human":
            st.markdown(f"🧑 **You:** {msg.content}")
        else:
            st.markdown(f"🤖 **Bot:** {msg.content}")


