import os.path
import datetime
import time
import random
import pandas as pd
import pathway as pw
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)

def fetch_emails_last_month(service):
    """Fetches emails using adaptive batching with exponential backoff."""
    
    today = datetime.date.today()
    thirty_days_ago = today - datetime.timedelta(days=30)
    query_date = thirty_days_ago.strftime('%Y/%m/%d')
    
    print(f"Fetching email IDs after {query_date}...")
    
    all_messages = []
    next_page_token = None
    
    # 1. Rapidly fetch all Message IDs first
    while True:
        results = service.users().messages().list(
            userId='me', 
            q=f'after:{query_date}',
            pageToken=next_page_token
        ).execute()
        
        batch_msgs = results.get('messages', [])
        all_messages.extend(batch_msgs)
        next_page_token = results.get('nextPageToken')
        
        if not next_page_token:
            break
            
    print(f"Found {len(all_messages)} emails. Starting high-speed download...")
    
    email_data = []
    
    # We track temporary state for the batch callback
    batch_results = {"failed_ids": [], "success_data": []}

    def batch_callback(request_id, response, exception):
        if exception:
            # If rate limit (429) or server error (5xx), we mark for retry
            if hasattr(exception, 'resp') and exception.resp.status in [429, 500, 503]:
                batch_results["failed_ids"].append(request_id)
            else:
                print(f"  [Error] Fatal error on email {request_id}: {exception}")
        else:
            # Success! Parse the email
            payload = response.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
            sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            snippet = response.get('snippet', '')
            
            batch_results["success_data"].append({
                "id": request_id,
                "date": date,
                "sender": sender,
                "subject": subject,
                "content": snippet 
            })

    # 2. Process in chunks with Retry Logic
    BATCH_SIZE = 50
    total_processed = 0
    
    # Iterate through the list of all message IDs in chunks
    for i in range(0, len(all_messages), BATCH_SIZE):
        # The IDs we want to process in this round
        current_batch_ids = [msg['id'] for msg in all_messages[i:i+BATCH_SIZE]]
        
        retry_attempt = 0
        
        # Retry Loop: Keep trying until this chunk is empty (all succeeded)
        while current_batch_ids:
            batch = service.new_batch_http_request(callback=batch_callback)
            
            # Clear previous batch results
            batch_results["failed_ids"] = []
            batch_results["success_data"] = []
            
            # Add requests to batch
            for msg_id in current_batch_ids:
                batch.add(service.users().messages().get(userId='me', id=msg_id), request_id=msg_id)
            
            try:
                batch.execute()
            except Exception as e:
                print(f"  [Critical] Entire batch failed: {e}")
                # If the whole batch explodes, assume all failed and need retry
                batch_results["failed_ids"] = current_batch_ids

            # 1. Save Successes
            email_data.extend(batch_results["success_data"])
            total_processed += len(batch_results["success_data"])
            
            # 2. Handle Failures
            failed = batch_results["failed_ids"]
            if not failed:
                break 
            else:
                # We had failures. Exponential Backoff time!
                retry_attempt += 1
                sleep_time = min(10, (2 ** retry_attempt)/10)
                
                print(f"  ⚠️ Rate limit hit on {len(failed)} emails. Backing off for {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                
                # Set the batch to ONLY the failed IDs for the next loop iteration
                current_batch_ids = failed
        
        print(f"  ...progress: {total_processed}/{len(all_messages)}")
            
    return email_data

def main():
    service = get_gmail_service()
    emails = fetch_emails_last_month(service)
    
    if not emails:
        print("No emails found.")
        return

    df = pd.DataFrame(emails)
    emails_table = pw.debug.table_from_pandas(df)
    
    print(f"\n--- Successfully loaded {len(df)} emails into Pathway ---")
    pw.debug.compute_and_print(emails_table.select(
        emails_table.date, 
        emails_table.sender, 
        emails_table.subject
    ), include_id=False)

if __name__ == '__main__':
    main()