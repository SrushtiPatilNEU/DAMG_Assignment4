import logging
import time
import os
import boto3
import requests
import pdfplumber
import tiktoken
import json
import sys
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import re
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# Initialize S3 client
def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

# Define DAG default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
dag = DAG(
    'nvidia_reports_processor',
    default_args=default_args,
    description='Process NVIDIA financial reports',
    schedule_interval=None,  # Set to None for manual triggering
    catchup=False
)

# Function to upload files to S3
def upload_to_s3(file_name, bucket, object_name):
    """Uploads a file to an S3 bucket with logging."""
    s3_client = get_s3_client()
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"✅ Uploaded {file_name} to s3://{bucket}/{object_name}")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to upload {file_name}: {str(e)}")
        return False

# Function to download PDFs
def download_pdf(url, filename):
    """Downloads a PDF from the given URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(filename, "wb") as pdf_file:
                for chunk in response.iter_content(1024):
                    pdf_file.write(chunk)
            logging.info(f"✅ Downloaded: {filename}")
            return filename
        else:
            logging.error(f"❌ Failed to download {url} - Status Code: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"❌ Error downloading {url}: {e}")
        return None

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        if not text.strip():
            logging.warning(f"⚠️ No text extracted from {pdf_path}.")
        return text.strip()
    except Exception as e:
        logging.error(f"❌ Error extracting text from {pdf_path}: {e}")
        return None

# Task 1: Scrape NVIDIA Financial Reports with BeautifulSoup
def scrape_nvidia_financial_reports(**kwargs):
    # Years to scrape from the dropdown
    years = ["2021", "2022", "2023", "2024", "2025"]
    # Set up Chrome options for headless mode
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = None

    try:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        logging.info("🚀 Accessing NVIDIA investor relations page...")
        driver.get("https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")
        driver.implicitly_wait(10)
        all_reports = []
        for year in years:
            try:
                logging.info(f"🔄 Selecting financial year: {year}...")
                # Select year from dropdown
                dropdown_element = WebDriverWait(driver, 20).until(
                    EC.element_to_be_clickable((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear"))
                )
                dropdown = Select(dropdown_element)
                dropdown.select_by_visible_text(year)
                time.sleep(3)
                # Expand all quarter sections
                logging.info(f"🔄 Expanding all quarters for {year}...")
                toggles = driver.find_elements(By.CSS_SELECTOR, ".accordion-toggle-icon.evergreen-icon-plus")
                for toggle in toggles:
                    try:
                        driver.execute_script("arguments[0].click();", toggle)
                        time.sleep(1)
                    except Exception as e:
                        logging.warning(f"⚠️ Could not expand quarter section for {year}: {str(e)}")
                # Scrape 10-K and 10-Q reports
                logging.info(f"🔍 Extracting 10-K and 10-Q reports for {year}...")
                links = driver.find_elements(By.CSS_SELECTOR, "a.evergreen-financial-accordion-attachment-PDF")
                for link in links:
                    link_text = link.text.strip()
                    url = link.get_attribute("href")
                    if url and url.endswith(".pdf") and ("10-K" in link_text or "10-Q" in link_text):
                        quarter = "Q1" if "/q1" in url.lower() else \
                                  "Q2" if "/q2" in url.lower() else \
                                  "Q3" if "/q3" in url.lower() else \
                                  "Q4" if "/q4" in url.lower() or "10-K" in link_text else "Unknown"
                        report = {
                            "year": year,
                            "quarter": quarter,
                            "type": link_text,
                            "url": url
                        }
                        all_reports.append(report)
                        logging.info(f"✅ Found: {report}")
            except Exception as e:
                logging.error(f"❌ Failed to process year {year}: {str(e)}")

        # Pass the results to the next task
        kwargs['ti'].xcom_push(key='scraped_reports', value=all_reports)
        return len(all_reports)
    except Exception as e:
        logging.error(f"❌ Error during scraping: {str(e)}")
        return 0
    finally:
        if driver:
            driver.quit()

# Task 2: Save and Upload to S3
def save_and_upload(**kwargs):
    """Saves extracted PDF content as markdown files and uploads to AWS S3."""
    reports = kwargs['ti'].xcom_pull(key='scraped_reports', task_ids='scrape_nvidia_financial_reports')
    
    if not reports:
        logging.warning("⚠️ No reports to process")
        return 0
    
    successfully_uploaded = []
    
    for report in reports:
        year = report["year"]
        quarter = report["quarter"]
        pdf_filename = f"{year}_{quarter}.pdf"
        # Download the PDF
        pdf_path = download_pdf(report["url"], pdf_filename)
        if not pdf_path:
            continue
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            continue  
        # Save extracted text to markdown
        md_filename = f"{year}_{quarter}.md"
        markdown_content = f"# {year} {quarter} Financial Report\n\n"
        markdown_content += f"**{report['type']}:** {report['url']}\n\n"
        markdown_content += f"### Extracted Content:\n\n{pdf_text}"
        try:
            with open(md_filename, "w", encoding="utf-8") as file:
                file.write(markdown_content)
            logging.info(f"✅ Saved file: {md_filename}")
            if upload_to_s3(md_filename, S3_BUCKET_NAME, f"{S3_PREFIX}/{md_filename}"):
                successfully_uploaded.append(md_filename)
            
            os.remove(md_filename)
            os.remove(pdf_filename)  # Clean up PDF after processing
            logging.info(f"🗑️ Deleted local files: {md_filename}, {pdf_filename}")
        except Exception as e:
            logging.error(f"❌ Error processing file {md_filename}: {e}")
    
    # Store the list of uploaded files for the next task
    kwargs['ti'].xcom_push(key='uploaded_files', value=successfully_uploaded)
    return len(successfully_uploaded)

# Task 3: Fetch Files from S3
def fetch_files_from_s3(**kwargs):
    """Fetches a list of files from S3."""
    s3_client = get_s3_client()
    files = []
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX)
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".md"):
                files.append(obj["Key"].split("/")[-1])
        logging.info(f"📂 Found {len(files)} markdown files in S3")
        
        # Check if we need to use files from previous task or all files from S3
        uploaded_files = kwargs['ti'].xcom_pull(key='uploaded_files', task_ids='save_and_upload')
        if uploaded_files and len(uploaded_files) > 0:
            logging.info(f"📂 Using {len(uploaded_files)} files from previous task")
            files = uploaded_files
    except Exception as e:
        logging.error(f"❌ Error fetching files from S3: {e}")
    
    kwargs['ti'].xcom_push(key='s3_files', value=files)
    return len(files)

# Helper functions for Pinecone
def get_file_content(file_key):
    """Retrieves the content of a file from S3."""
    s3_client = get_s3_client()
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_PREFIX}/{file_key}")
        content = response["Body"].read().decode("utf-8")
        if not content.strip():
            logging.warning(f"⚠️ Warning: {file_key} is empty.")
            return None
        return content
    except Exception as e:
        logging.error(f"❌ Error retrieving {file_key}: {e}")
        return None

def openai_token_count(text):
    """Returns the number of tokens in a given text using OpenAI's tokenizer."""
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer
    return len(encoding.encode(text))  # Return the token count

# Chunking methods
def section_based_chunking(markdown_text):
    sections = re.split(r'(?=^#+ )', markdown_text, flags=re.MULTILINE)
    return [section.strip() for section in sections if section.strip()]

def table_based_chunking(markdown_text):
    tables = re.findall(r'(\|[^\n]+\|\n)+', markdown_text)
    return tables

def sliding_window_chunking(text, window_size=500, stride=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), stride):
        chunk = ' '.join(words[i:i+window_size])
        chunks.append(chunk)
    return chunks

# Embedding function
def embed_text(text):
    """Generate OpenAI embeddings for a given text chunk."""
    openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    return openai_embeddings.embed_query(text)

# Task 4: Process and Store in Pinecone
def process_and_store_in_pinecone(**kwargs):
    """Processes files: fetches from S3, applies chunking, embeds, and stores in Pinecone."""
    files = kwargs['ti'].xcom_pull(key='s3_files', task_ids='fetch_files_from_s3')
    
    if not files:
        logging.warning("⚠️ No files to process for Pinecone")
        return 0
    
    MAX_REQUEST_SIZE = 2 * 1024 * 1024  # 2MB
    
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if index exists
        try:
            index_list = pc.list_indexes().names()
            if INDEX_NAME not in index_list:
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT)
                )
                # Wait for the index to be initialized
                while not pc.describe_index(INDEX_NAME).status['ready']:
                    time.sleep(1)
        except Exception as e:
            logging.error(f"❌ Error with Pinecone index creation: {str(e)}")
            return 0
        
        index = pc.Index(INDEX_NAME, pool_threads=10, timeout=60)
        
        total_chunks = 0
     
        for file_key in files:
            logging.info(f"📂 Processing file: {file_key}")
     
            text_content = get_file_content(file_key)
            if not text_content:
                continue
     
            filename = file_key.replace(".md", "")
            parts = filename.split("_")
            year = parts[0] if len(parts) > 1 else "Unknown"
            quarter = parts[1] if len(parts) > 1 else "Unknown"
     
            # Apply all three chunking methods
            chunking_methods = {
                "Section": section_based_chunking(text_content),
                "Table": table_based_chunking(text_content),
                "SlidingWindow": sliding_window_chunking(text_content)
            }
     
            for method_name, chunks in chunking_methods.items():
                logging.info(f"🔹 Using {method_name} Chunking: {len(chunks)} chunks from {file_key}")
                total_chunks += len(chunks)
     
                batch_vectors = []
                current_batch_size = 0
     
                for i, chunk in enumerate(chunks):
                    embedding = embed_text(chunk)
                    vector_id = f"{year}-{quarter}-{method_name}-{i}"
     
                    metadata = {
                        "year": year,
                        "quarter": quarter,
                        "chunk_type": method_name,
                        "chunk_index": i,
                        "text": chunk
                    }
     
                    # Estimate the size of the vector entry
                    vector_entry = {
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                    vector_size = sys.getsizeof(json.dumps(vector_entry))
     
                    # Check if adding this vector exceeds the 2MB limit
                    if current_batch_size + vector_size > MAX_REQUEST_SIZE:
                        # Upsert the current batch
                        try:
                            index.upsert(vectors=batch_vectors)
                            logging.info(f"✅ Upserted batch of {len(batch_vectors)} vectors")
                        except Exception as e:
                            logging.error(f"❌ Error upserting batch: {str(e)}")
                        batch_vectors = []
                        current_batch_size = 0
     
                    batch_vectors.append(vector_entry)
                    current_batch_size += vector_size
     
                # Final batch upload
                if batch_vectors:
                    try:
                        index.upsert(vectors=batch_vectors)
                        logging.info(f"✅ Upserted final batch of {len(batch_vectors)} vectors")
                    except Exception as e:
                        logging.error(f"❌ Error upserting final batch: {str(e)}")
     
            logging.info(f"✅ Stored all chunking methods for {file_key} in Pinecone")
    except Exception as e:
        logging.error(f"❌ Error in Pinecone processing: {str(e)}")
        return 0
    
    return total_chunks

# Create task instances
scrape_task = PythonOperator(
    task_id='scrape_nvidia_financial_reports',
    python_callable=scrape_nvidia_financial_reports,
    provide_context=True,
    dag=dag,
)

upload_task = PythonOperator(
    task_id='save_and_upload',
    python_callable=save_and_upload,
    provide_context=True,
    dag=dag,
)

fetch_files_task = PythonOperator(
    task_id='fetch_files_from_s3',
    python_callable=fetch_files_from_s3,
    provide_context=True,
    dag=dag,
)

process_pinecone_task = PythonOperator(
    task_id='process_and_store_in_pinecone',
    python_callable=process_and_store_in_pinecone,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
scrape_task >> upload_task >> fetch_files_task >> process_pinecone_task
