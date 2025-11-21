import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def reset_database():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ No DATABASE_URL found in .env")
        return

    print("⚠️  WARNING: This will PERMANENTLY DELETE all email data in your database.")
    confirm = input("Are you sure you want to drop the 'emails' table? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return

    try:
        engine = create_engine(db_url)
        # Use a connection block to ensure the DROP command is committed
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS emails"))
        print("✅ Success! The 'emails' table has been dropped.")
        print("Run 'python gmail_rag.py' to re-ingest your data.")
    except Exception as e:
        print(f"❌ Error resetting database: {e}")

if __name__ == "__main__":
    reset_database()