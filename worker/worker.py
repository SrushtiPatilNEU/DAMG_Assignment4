import redis
import json
import time
import litellm
import os
from dotenv import load_dotenv
import tempfile
import shutil
import logging
from fastapi import FastAPI
import uvicorn
from threading import Thread
from google.cloud import storage


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# FastAPI application setup
app = FastAPI()

# Set up Redis client for communication
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

try:
    if redis_client.ping():
        logger.info("✅ Successfully connected to Redis!")
    else:
        logger.error("❌ Redis connection failed!")
except redis.ConnectionError as e:
    logger.error(f"❌ Redis connection error: {e}")

STREAM_NAME = "llm_requests"

# LLM model configurations with appropriate keys and provider info
LLM_MODELS = {
    "GPT-4o": {"model": "gpt-4o", "api_key": os.getenv("GPT4o_API_KEY")},  
    "Gemini-Flash": {"model": "gemini/gemini-2.0-flash-exp", "api_key": os.getenv("GEMINI_API_KEY"), "provider": "google"},
    "DeepSeek": {"model": "deepseek/deepseek-chat", "api_key": os.getenv("DEEPSEEK_API_KEY"), "provider": "deepseek"},
    "Claude": {"model": "claude-3-5-sonnet-20240620", "api_key": os.getenv("CLAUDE_API_KEY")},  
    "Grok": {"model": "xai/grok-2-1212", "api_key": os.getenv("GROK_API_KEY"), "provider": "grok"}
}

'''

def setup_google_credentials():
    """Set up Google Cloud credentials from a GCP bucket."""
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    credentials_filename = os.getenv("GCP_CREDENTIALS_JSON_FILENAME")
    
    if not bucket_name or not credentials_filename:
        logger.error("GCP_BUCKET_NAME or GCP_CREDENTIALS_JSON_FILENAME not found in environment variables.")
        return None

    try:
        # Create credentials directory if it doesn't exist
        credentials_dir = "/app/credentials"
        os.makedirs(credentials_dir, exist_ok=True)
        
        # Path to credentials file
        credentials_path = f"{credentials_dir}/google-credentials.json"
        
        # Initialize the Google Cloud Storage client
        storage_client = storage.Client()
        
        # Download the credentials file from the bucket
        logger.info(f"Downloading Google credentials from bucket {bucket_name} to {credentials_path}")
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(credentials_filename)
        blob.download_to_filename(credentials_path)

        # Set the environment variable for Google authentication
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        logger.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to {credentials_path}")
        
        return credentials_path
    except Exception as e:
        logger.error(f"Failed to set up Google credentials: {str(e)}")
        return None
'''

def call_llm(llm_name, prompt):
    """Send a request to the selected LLM model using LiteLLM."""
    model_info = LLM_MODELS.get(llm_name)

    if not model_info or not model_info["api_key"]:
        return f"❌ Error: API key missing for {llm_name}"

    try:
        # Handle Gemini model specifically
        if llm_name == "Gemini-Flash":
            '''
            # Make sure Google credentials are set up
            credentials_path = setup_google_credentials()
            if not credentials_path:
                return f"❌ Error: Failed to set up Google credentials for {llm_name}"'
            '''
                
            # Use the standard Google authentication method
            response = litellm.completion(
                model=model_info["model"],
                messages=[{"role": "user", "content": prompt}],
                api_key=model_info["api_key"],
                provider=model_info["provider"]
            )
        # Handle other models
        elif "provider" in model_info:
            response = litellm.completion(
                model=model_info["model"],
                messages=[{"role": "user", "content": prompt}],
                api_key=model_info["api_key"],
                provider=model_info["provider"]
            )
        else:
            response = litellm.completion(
                model=model_info["model"],
                messages=[{"role": "user", "content": prompt}],
                api_key=model_info["api_key"]
            )

        return response['choices'][0]['message']['content']

    except Exception as e:
        error_msg = f"❌ Error calling {llm_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def process_redis_messages():
    """Function to process Redis messages continuously in a background thread."""
    logger.info("Worker started, waiting for messages...")

    while True:
        try:
            # Read messages from the Redis stream
            messages = redis_client.xread({STREAM_NAME: "0"}, count=1, block=1000)

            for stream, message_list in messages:
                for msg_id, msg_data in message_list:
                    logger.info(f"🔍 Received raw message: {msg_data}")

                    if "data" not in msg_data:
                        logger.warning(f"⚠️ Skipping message {msg_id}, missing 'data' field.")
                        continue

                    try:
                        msg = json.loads(msg_data["data"])
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Skipping malformed JSON: {msg_data}")
                        continue

                    if "task_id" not in msg or "type" not in msg or "pdf_name" not in msg or "llm" not in msg or "content" not in msg:
                        logger.warning(f"⚠️ Skipping invalid message: {msg}")
                        continue

                    task_id = msg["task_id"]
                    content = msg["content"].strip()

                    if not content:
                        logger.warning(f"⚠️ Skipping {task_id}: No content provided.")
                        redis_client.set(f"response:{task_id}", "❌ Error: Document is empty, cannot summarize.")
                        continue

                    logger.info(f"🚀 Processing Task: {task_id} - Type: {msg['type']} - Model: {msg['llm']}")

                    # Prepare prompt based on task type (summarize or QA)
                    if msg["type"] == "summarize":
                        prompt = f"Summarize this document:\n\n{content}"
                    elif msg["type"] == "qa":
                        prompt = f"Answer this question: {msg['question']} based on the following document:\n\n{content}"
                    else:
                        logger.warning(f"⚠️ Unknown task type: {msg['type']}")
                        continue

                    # Call the LLM with the appropriate model and prompt
                    response = call_llm(msg["llm"], prompt)

                    # Store the response in Redis
                    redis_client.set(f"response:{task_id}", response)  # ✅ Store result
                    logger.info(f"✅ Task {task_id} completed and stored in Redis")

                    # Mark the task as processed by deleting it from the Redis stream
                    redis_client.xdel(STREAM_NAME, msg_id)  # ✅ Delete processed task

            # Sleep briefly before checking for new messages
            time.sleep(1)

        except Exception as e:
            logger.error(f"❌ Worker Error: {str(e)}")
            time.sleep(2)

# Setup Google credentials at startup
'''
if "Gemini-Flash" in LLM_MODELS:
    setup_google_credentials()'
'''

# Run the Redis worker in a background thread
thread = Thread(target=process_redis_messages)
thread.daemon = True
thread.start()

# Make sure you bind to 0.0.0.0 and port 8080
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
