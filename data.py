
from app import db

def seed_data():
    db.kb.delete_many({})
    db.user_login_history.delete_many({})
    db.user_purchase_history.delete_many({})
    db.user_logs_history.delete_many({})
    db.chat_history.delete_many({})
    db.users.delete_many({})

    # Add sample users
    db.users.insert_many([
        {"username": "alice", "password": "1234", "user_id": "user_123"},
        {"username": "bob", "password": "5678", "user_id": "user_456"},
    ])

    db.kb.insert_many([
        {"question": "How do I reset my password?",
         "answer": "Click on 'Forgot Password' on the login page and follow the steps."},
        {"question": "What should I do if my laptop doesn’t turn on?",
         "answer": "Please check the power connection and try holding the power button for 10 seconds."},
        {"question": "My payment failed but money deducted.",
         "answer": "Please provide your transaction ID. We will check and process a refund if applicable."}
    ])

    
    db.user_login_history.insert_many([
        {"user_id": "user_123", "login_time": "2025-09-01T08:12:00", "location": "Mumbai", "email": "alice@example.com"},
        {"user_id": "user_456", "login_time": "2025-09-02T12:45:00", "location": "Delhi", "email": "bob@example.com"}
    ])


    db.user_purchase_history.insert_many([
        {"user_id": "user_123", "product_name": "Lenovo ThinkPad X1", "purchase_date": "2025-08-12", "amount": 1200},
        {"user_id": "user_456", "product_name": "Wireless Mouse", "purchase_date": "2025-07-03", "amount": 25}
    ])


    db.user_logs_history.insert_many([
        {"user_id": "user_123", "issue": "Laptop won’t start", "resolution": "Pending", "created_at": "2025-09-01"},
        {"user_id": "user_456", "issue": "Order delayed", "resolution": "Resolved", "created_at": "2025-08-10"}
    ])

    print("✅ Database seeded with sample data")

if __name__ == "__main__":
    seed_data()
