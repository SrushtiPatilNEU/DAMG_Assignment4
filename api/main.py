import os
import json
import boto3
import base64
import io
import uvicorn
import redis
import fitz  # PyMuPDF for PDF parsing
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
from openSourcePdf import extract_data, save_to_md

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=S3_REGION,
)

def upload_to_s3(file_content: bytes, folder: str, filename: str, content_type: str) -> None:
    s3_path = f"{folder}/{filename}"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_path,
            Body=file_content,
            ContentType=content_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 Upload Failed: {str(e)}")
    
def extract_text_from_pdf(pdf_file_io: io.BytesIO):
    """Extracts only text from the PDF (no images)."""
    try:
        doc = fitz.open(stream=pdf_file_io, filetype="pdf")
        extracted_text = "\n".join([page.get_text("text") for page in doc])
        return extracted_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Extraction Failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI()

# Initialize Redis
##redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
#redis_client = redis.Redis(host="host.docker.internal", port=6379, decode_responses=True)
##redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "host.docker.internal"), port=int(os.getenv("REDIS_PORT", 6379)), decode_responses=True)

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

# Storage directories
UPLOAD_DIR = "uploads"
MARKDOWN_DIR = "markdowns"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MARKDOWN_DIR, exist_ok=True)


@app.get("/")
async def home():
    return {"message": "AI Document Processing API"}

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_file_io = io.BytesIO(contents)

        # ✅ Extract only text from the PDF
        extracted_text = extract_text_from_pdf(pdf_file_io)

        if extracted_text:
            # ✅ Convert PDF filename to `.md` (same name as original PDF)
            md_filename = file.filename.rsplit(".", 1)[0] + ".md"  # ✅ Keep original name, change to `.md`
            markdown_content = f"# Extracted Text from {file.filename}\n\n" + extracted_text

            # ✅ Upload Markdown file to S3
            upload_to_s3(markdown_content.encode(), "new_upload/markdown", md_filename, "text/markdown")

            return {
                "filename": md_filename,  # ✅ Now filename matches original PDF name
                "s3_url": f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/new_upload/markdown/{md_filename}",
                "message": "✅ PDF processed and stored as Markdown in S3."
            }
        else:
            raise HTTPException(status_code=400, detail="❌ No Extracted Text Found in the PDF.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/select_pdfcontent/")
async def get_markdowns():
    """Retrieve the list of Markdown files stored in S3."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="")
        if "Contents" not in response:
            return {"markdowns": []} # No files found
        markdown_files = [
            obj["Key"].split("/")[-1] for obj in response["Contents"] if obj["Key"].endswith(".md")
        ]
        return {"markdowns": markdown_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to list Markdown files from S3: {str(e)}")
    
'''
@app.get("/download_markdown/{filename}")
async def download_markdown(filename: str):
    """Download the extracted markdown file."""
    file_path = os.path.join(MARKDOWN_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
'''

@app.get("/download_markdown/{filename}")
async def download_markdown(filename: str):
    """Download the extracted markdown file from S3."""
    try:
        file_path = f"new_upload/markdown/{filename}"  # S3 path

        # Download the file from S3 to a local temporary file
        s3_client.download_file(S3_BUCKET_NAME, file_path, filename)

        return FileResponse(filename, filename=filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Failed to download file from S3: {str(e)}")


def read_file_content(file_path):
    """Reads and returns the content of a markdown file."""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"❌ Error: File {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        raise HTTPException(status_code=400, detail=f"❌ Error: {file_path} is empty.")

    return content

@app.post("/summarize/")
async def summarize(pdf_name: str = Form(...), llm: str = Form(...)):
    """Send a summarization request to the queue with the file content."""
    try:
        # Download the file from S3 to a local temporary file
        s3_client.download_file(S3_BUCKET_NAME, f"new_upload/markdown/{pdf_name}", pdf_name)

        # Read file content
        with open(pdf_name, "r", encoding="utf-8") as file:
            file_content = file.read()

        if not file_content.strip():
            raise HTTPException(status_code=400, detail=f"❌ Error: {pdf_name} is empty.")

        task_id = f"task-{os.urandom(4).hex()}"

        # Send file content to Redis for processing
        task_data = json.dumps({
            "task_id": task_id,
            "type": "summarize",
            "pdf_name": pdf_name,
            "llm": llm,
            "content": file_content  # Sending the actual text
        })

        redis_client.xadd(STREAM_NAME, {"data": task_data})

        # Remove the temporary file
        os.remove(pdf_name)

        return {"task_id": task_id, "message": "✅ Summarization request added"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")

@app.post("/ask_question/")
async def ask_question(pdf_name: str = Form(...), llm: str = Form(...), question: str = Form(...)):
    """Send a question-answering request with document content to Redis."""
    try:
        # Download the file from S3 to a local temporary file
        s3_client.download_file(S3_BUCKET_NAME, f"new_upload/markdown/{pdf_name}", pdf_name)

        # Read file content
        with open(pdf_name, "r", encoding="utf-8") as file:
            content = file.read()

        task_id = f"task-{os.urandom(4).hex()}"

        task_data = json.dumps({
            "task_id": task_id,
            "type": "qa",
            "pdf_name": pdf_name,
            "llm": llm,
            "question": question,
            "content": content  # Send actual content
        })

        redis_client.xadd(STREAM_NAME, {"data": task_data})

        # Remove the temporary file
        os.remove(pdf_name)

        return {"task_id": task_id, "message": "✅ Q&A request added"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")


'''
@app.post("/summarize/")
async def summarize(pdf_name: str = Form(...), llm: str = Form(...)):
    """Send a summarization request to the queue with the file content."""
    file_path = os.path.join(MARKDOWN_DIR, pdf_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"❌ Error: File {pdf_name} not found.")

    # ✅ Read file content
    with open(file_path, "r", encoding="utf-8") as file:
        file_content = file.read()

    if not file_content.strip():
        raise HTTPException(status_code=400, detail=f"❌ Error: {pdf_name} is empty.")

    task_id = f"task-{os.urandom(4).hex()}"

    # ✅ Send file content to Redis for processing
    task_data = json.dumps({
        "task_id": task_id,
        "type": "summarize",
        "pdf_name": pdf_name,
        "llm": llm,
        "content": file_content  # ✅ Sending the actual text
    })

    redis_client.xadd(STREAM_NAME, {"data": task_data})
    return {"task_id": task_id, "message": "✅ Summarization request added"}



@app.post("/ask_question/")
async def ask_question(pdf_name: str = Form(...), llm: str = Form(...), question: str = Form(...)):
    """Send a question-answering request with document content to Redis."""
    file_path = os.path.join(MARKDOWN_DIR, pdf_name)
    content = read_file_content(file_path)  # ✅ Read file content before sending to Redis

    task_id = f"task-{os.urandom(4).hex()}"

    task_data = json.dumps({
        "task_id": task_id,
        "type": "qa",
        "pdf_name": pdf_name,
        "llm": llm,
        "question": question,
        "content": content  # ✅ Send actual content
    })

    redis_client.xadd(STREAM_NAME, {"data": task_data})
    return {"task_id": task_id, "message": "✅ Q&A request added"}
'
'''

@app.get("/get_result/{task_id}")
async def get_result(task_id: str):
    """Fetch the result of an AI task."""
    result = redis_client.get(f"response:{task_id}")
    if result:
        return {"result": result}
    return JSONResponse(content={"message": "Processing..."}, status_code=202)
