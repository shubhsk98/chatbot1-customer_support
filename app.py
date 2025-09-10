# app.py
import os
import pymongo
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------------------------
# User Authentication
# ---------------------------
users_collection = db["users"]
chat_collection = db["chat_history"]
logs_collection = db["user_logs"]

# ==============================
# User Functions
# ==============================

def get_user(username, password):
    return users_collection.find_one({"username": username, "password": password})

def create_user(username, password):
    user_id = str(uuid.uuid4())  # Generates a unique user ID
    users_collection.insert_one({
        "user_id": user_id,
        "username": username,
        "password": password,
        "created_at": datetime.now().isoformat(),
        "logins": 1
    })
    return user_id

# ---------------------------
# User Data Fetch Functions
# ---------------------------
def fetch_user_data(user_id):
    logins = list(db.user_login_history.find({"user_id": user_id}, {"_id": 0}))
    purchases = list(db.user_purchase_history.find({"user_id": user_id}, {"_id": 0}))
    logs = list(db.user_logs_history.find({"user_id": user_id}, {"_id": 0}))
    return {"logins": logins, "purchases": purchases, "logs": logs}

# ---------------------------
# Insert Functions
# ---------------------------
def insert_user_login(user_id, login_time, location, email):
    db.user_login_history.insert_one({
        "user_id": user_id,
        "login_time": login_time,
        "location": location,
        "email": email
    })


def insert_user_purchase(user_id, product_name, purchase_date, amount):
    db.user_purchase_history.insert_one({
        "user_id": user_id,
        "product_name": product_name,
        "purchase_date": purchase_date,
        "amount": amount
    })

def insert_user_log(user_id, issue, resolution, created_at):
    db.user_logs_history.insert_one({
        "user_id": user_id,
        "issue": issue,
        "resolution": resolution,
        "created_at": created_at
    })

# ---------------------------
# Knowledge Base (FAQ) Fetch
# ---------------------------
def get_kb_docs():
    kb = list(db.kb.find({}, {"_id": 0}))
    return kb

# ---------------------------
# Chat History Functions
# ---------------------------
def save_chat_history(user_id, query, response):
    db.chat_history.insert_one({
        "user_id": user_id,
        "query": query,
        "response": response,
        "created_at": datetime.now().isoformat()
    })

def get_chat_history(user_id):
    chats = list(db.chat_history.find({"user_id": user_id}, {"_id": 0}))
    return chats

