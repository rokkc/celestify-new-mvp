import os
import datetime
import time
import random
import asyncio
import pandas as pd
import numpy as np
import pathway as pw
import google.generativeai as genai
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dateutil import parser as date_parser
from tzlocal import get_localzone

# 1. Load Secrets
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
DB_URL = os.getenv("DATABASE_URL")

# --- CONFIGURATION ---
DEFAULT_LIMIT = 50
USER_TIMEZONE = get_localzone()

# --- DATABASE HELPERS ---

def get_db_engine():
    if not DB_URL:
        print("⚠️ No DATABASE_URL found. SQL features will fail.")
        return None
    return create_engine(DB_URL)

def get_existing_ids(engine):
    if not engine: return set()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT id FROM emails"))
            return set(row[0] for row in result)
    except Exception:
        return set()

def get_latest_date_from_db(engine):
    if not engine: return None
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT MAX(date) FROM emails"))
            latest = result.scalar()
            if latest:
                # Ensure we handle the format returned by Postgres
                last_date = pd.to_datetime(latest, utc=True) - pd.Timedelta(days=1)
                return last_date.strftime('%Y/%m/%d')
    except Exception:
        return None
    return None

def save_to_db(df, engine):
    if not engine or df.empty: return
    print(f"💾 Saving {len(df)} new emails to RDS...")
    
    # --- FIX: Force UTC conversion to handle Gmail's timezones ---
    df['date'] = pd.to_datetime(df['date'], utc=True)
    # -------------------------------------------------------------
    
    df.to_sql('emails', engine, if_exists='append', index=False)

# --- GMAIL INGESTION ---

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def fetch_emails_since(service, start_date=None):
    if not start_date:
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=30)).strftime('%Y/%m/%d')
    
    print(f"Fetching email IDs after {start_date}...")
    all_messages = []
    next_page_token = None
    
    while True:
        results = service.users().messages().list(
            userId='me', q=f'after:{start_date}', pageToken=next_page_token
        ).execute()
        batch_msgs = results.get('messages', [])
        all_messages.extend(batch_msgs)
        next_page_token = results.get('nextPageToken')
        if not next_page_token: break
            
    if not all_messages:
        print("✅ No new emails found.")
        return []

    print(f"Found {len(all_messages)} potential emails. Downloading content...")
    email_data = []
    batch_results = {"failed_ids": [], "success_data": []}

    def batch_callback(request_id, response, exception):
        if exception:
            if hasattr(exception, 'resp') and exception.resp.status in [429, 500, 503]:
                batch_results["failed_ids"].append(request_id)
            else:
                print(f"  [Error] Fatal error on email {request_id}")
        else:
            payload = response.get('payload', {})
            headers = payload.get('headers', [])
            snippet = response.get('snippet', '')
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            raw_date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            
            try:
                date_obj = date_parser.parse(raw_date, fuzzy=True)
            except:
                # FIX: Use UTC now to avoid mixed-timezone errors
                date_obj = datetime.datetime.now(datetime.timezone.utc)
            
            full_text = f"From: {sender}\nDate: {raw_date}\nSubject: {subject}\nContent: {snippet}"
            
            batch_results["success_data"].append({
                "id": request_id,
                "text": full_text,
                "sender": sender,
                "subject": subject,
                "date": date_obj
            })

    BATCH_SIZE = 50
    total_processed = 0
    
    for i in range(0, len(all_messages), BATCH_SIZE):
        current_batch_ids = [msg['id'] for msg in all_messages[i:i+BATCH_SIZE]]
        retry_attempt = 0
        
        while current_batch_ids:
            batch = service.new_batch_http_request(callback=batch_callback)
            batch_results["failed_ids"] = []
            batch_results["success_data"] = []
            
            for msg_id in current_batch_ids:
                batch.add(service.users().messages().get(userId='me', id=msg_id), request_id=msg_id)
            
            try:
                batch.execute()
            except Exception:
                batch_results["failed_ids"] = current_batch_ids

            email_data.extend(batch_results["success_data"])
            total_processed += len(batch_results["success_data"])
            
            failed = batch_results["failed_ids"]
            if not failed:
                break 
            else:
                retry_attempt += 1
                time.sleep(min(10, (2 ** retry_attempt)/10) + random.uniform(0, 0.1))
                current_batch_ids = failed
        
        print(f"  ...progress: {total_processed}/{len(all_messages)}")
            
    return email_data

# --- AI LOGIC ---

@pw.udf
async def embed_text(text: str) -> np.ndarray:
    if not text: return np.zeros(768)
    def _blocking_api_call():
        try:
            clean_text = text.replace("\n", " ")
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=clean_text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'])
        except Exception:
            return np.zeros(768)
    return await asyncio.to_thread(_blocking_api_call)

def get_time_and_limit_sql(question):
    """
    Generates SQL based on TIME, DATE, and QUANTITY constraints only.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    now_local = datetime.datetime.now(USER_TIMEZONE)
    
    prompt = f"""
    You are a SQL Query Generator specialized in Time and Quantity filtering.
    Current Local Time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}
    
    Table Schema: 'emails'
    Column: 'date' (datetime)
    
    Goal: Generate a SQL query to retrieve emails based ONLY on TIME and QUANTITY constraints.
    
    STRICT RULES:
    1. FILTER ONLY BY TIME (WHERE) AND QUANTITY (LIMIT). 
    2. Do NOT add WHERE clauses for 'sender', 'subject', or 'content'.
    3. "Recent" generally means the last 4-7 days unless a specific count (e.g. "top 5") is requested.
    4. ALWAYS use 'SELECT *'. Do NOT use 'SELECT COUNT(*)'. Even if the user asks "how many", select the actual emails so we can count them in Python.
    5. Use PostgreSQL syntax for date logic (e.g., NOW() - INTERVAL '7 days').
    
    SCENARIOS:
    - "10 most recent emails" -> ORDER BY date DESC LIMIT 10
    - "First 5 emails after Nov 5th" -> WHERE date > '202X-11-05' ORDER BY date ASC LIMIT 5
    - "Last 3 emails from October" -> WHERE date >= '202X-10-01' AND date < '202X-11-01' ORDER BY date DESC LIMIT 3
    - "Summarize my emails" (Vague) -> ORDER BY date DESC LIMIT {DEFAULT_LIMIT}
    - "What were my recent emails from Paula's Choice about?" -> WHERE date >= NOW() - INTERVAL '4 days' ORDER BY date DESC LIMIT {DEFAULT_LIMIT}
    - "How many emails last week?" -> WHERE date >= NOW() - INTERVAL '7 days' ORDER BY date DESC LIMIT {DEFAULT_LIMIT}
    
    6. Output ONLY the raw SQL query. No markdown.

    7. Use the context of the question to decide on specific values for the SQL query. "Recent" could mean very different time ranges depending on the question.

    8. DO NOT filter the emails at all if the question doesn't indicate a need to filter (ex. there's no need to filter by time or quantity if the user asks "What did my English teacher say about the final essay?" but there is a need to filter if the user asks "Summarize my recent emails.")
    
    User Question: "{question}"
    """
    
    response = model.generate_content(prompt)
    clean_sql = response.text.replace("```sql", "").replace("```", "").strip()
    
    # Safety check
    if "sender" in clean_sql.lower() or "subject" in clean_sql.lower():
        print("⚠️ LLM attempted to filter by content. Reverting to default recent.")
        return f"SELECT * FROM emails ORDER BY date DESC LIMIT {DEFAULT_LIMIT}"
        
    return clean_sql

def get_answer(context, question):
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = (
        "You are an expert Email Analyst.\n"
        "You have been provided with a list of emails filtered by TIME and COUNT.\n"
        "Now, answer the user's specific question based on these emails.\n"
        "If the user asked for specific senders or topics, find them within this provided list.\n\n"
        f"EMAILS FOUND:\n{context}\n\n"
        f"QUESTION: {question}\n"
    )
    response = model.generate_content(prompt)
    return response.text

# --- MAIN EXECUTION ---

def main():
    print(f"🕒 System Timezone: {USER_TIMEZONE}")
    engine = get_db_engine()

    # 1. Ingest New Emails
    existing_ids = get_existing_ids(engine)
    last_sync = get_latest_date_from_db(engine)
    service = get_gmail_service()
    
    fetched = fetch_emails_since(service, start_date=last_sync)
    new_emails = [e for e in fetched if e['id'] not in existing_ids]
    
    if new_emails:
        save_to_db(pd.DataFrame(new_emails), engine)
    else:
        print("✅ DB is up to date.")

    # 2. Interactive Loop
    while True:
        question = input("\nAsk your inbox (or 'q' to quit): ")
        if question.lower() in ['q', 'quit']: break
            
        print("🤔 Applying time & limit filters...")
        
        try:
            # A. Generate Time/Limit SQL
            sql_query = get_time_and_limit_sql(question)
            print(f"  Executing SQL: {sql_query}")
            
            # B. Execute SQL
            df_results = pd.read_sql(sql_query, engine)
            print(f"  Found {len(df_results)} emails.")
            
            if df_results.empty:
                print("  No emails matched your criteria.")
                continue

            # C. Hybrid Search Logic
            final_context_rows = []
            
            # If few results, use them all. If many, use Vector Search.
            if len(df_results) <= DEFAULT_LIMIT:
                print("  Count is low; using all retrieved emails for context.")
                final_context_rows = df_results
            else:
                print(f"  Count is high ({len(df_results)}); refining with Vector Search...")
                
                # --- FIX: Rename 'id' to 'doc_id' to avoid Pathway column collision ---
                if 'id' in df_results.columns:
                    df_results = df_results.rename(columns={'id': 'doc_id'})
                # --------------------------------------------------------------------
                
                t = pw.debug.table_from_pandas(df_results)
                enriched = t.select(
                    *pw.this.without(pw.this.text),
                    text=pw.this.text,
                    vector=embed_text(pw.this.text)
                )
                enriched_df = pw.debug.table_to_pandas(enriched)
                
                q_embed = genai.embed_content(
                    model="models/text-embedding-004",
                    content=question,
                    task_type="retrieval_query"
                )['embedding']
                
                vectors = np.stack(enriched_df['vector'].values)
                scores = np.dot(vectors, q_embed)
                
                top_indices = np.argsort(scores)[-7:][::-1]
                final_context_rows = enriched_df.iloc[top_indices]

            # D. Format for LLM
            context_text = ""
            for idx, row in final_context_rows.iterrows():
                d_str = row['date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                context_text += f"---\nDate: {d_str}\nFrom: {row['sender']}\nSubject: {row['subject']}\nContent: {row['text']}\n"
            
            print("🤖 Answering question...")
            answer = get_answer(context_text, question)
            print(f"\nANSWER:\n{answer}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

if __name__ == '__main__':
    main()