import uuid
import fitz
import re
import asyncio
import os
import signal
import sys
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from search import Index

app = FastAPI(title="Intelligent PDF Document Retrieval System")

index = Index()

MAX_PDFS = int(os.getenv("MAX_PDFS", "10"))

indexing_progress = {
    "status": "idle",
    "current": 0,
    "total": 0,
    "current_file": "",
    "message": ""
}
def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return clean_text(text)

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="<h1>Error: Template file not found</h1>", status_code=500)

async def process_pdfs_background(file_data: List[Dict]):
    """Background task to process PDFs and update progress"""
    global indexing_progress
    
    total = len(file_data)
    
    if total == 0:
        indexing_progress = {
            "status": "error",
            "current": 0,
            "total": 0,
            "current_file": "",
            "message": "No valid PDF files found"
        }
        return
    
    indexing_progress = {
        "status": "processing",
        "current": 0,
        "total": total,
        "current_file": "",
        "message": "Starting indexing..."
    }
    
    documents_to_add = []
    
    try:
        for i, file_info in enumerate(file_data):
            filename = file_info["filename"]
            content = file_info["content"]
            
            indexing_progress["current"] = i
            indexing_progress["current_file"] = filename
            indexing_progress["message"] = f"Processing {filename} ({i+1}/{total})..."
            
            text = extract_text(content)
            doc_id = str(uuid.uuid4())
            documents_to_add.append({"id": doc_id, "name": filename, "text": text})
            
            await asyncio.sleep(0.05)
        
        indexing_progress["message"] = f"Building search index for {len(documents_to_add)} documents..."
        indexing_progress["current"] = total
        
        added_count = index.add_documents_batch(documents_to_add)
        
        indexing_progress = {
            "status": "complete",
            "current": total,
            "total": total,
            "current_file": "",
            "message": f"Successfully indexed {added_count} PDF(s)!"
        }
    except Exception as e:
        indexing_progress = {
            "status": "error",
            "current": indexing_progress.get("current", 0),
            "total": total,
            "current_file": indexing_progress.get("current_file", ""),
            "message": f"Error: {str(e)}"
        }

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    global indexing_progress
    indexing_progress = {
        "status": "idle",
        "current": 0,
        "total": 0,
        "current_file": "",
        "message": ""
    }
    
    current_doc_count = len(index.docs)
    
    file_data = []
    for f in files:
        if f.filename and f.filename.lower().endswith(".pdf"):
            content = await f.read()
            file_data.append({"filename": f.filename, "content": content})
    
    total_after_upload = current_doc_count + len(file_data)
    if total_after_upload > MAX_PDFS:
        available_slots = MAX_PDFS - current_doc_count
        if available_slots <= 0:
            return JSONResponse({
                "status": "error",
                "message": f"Maximum limit of {MAX_PDFS} PDFs reached. Please remove some documents before uploading new ones.",
                "current_count": current_doc_count,
                "max_allowed": MAX_PDFS
            }, status_code=400)
        else:
            return JSONResponse({
                "status": "error",
                "message": f"Maximum limit of {MAX_PDFS} PDFs exceeded. You can only upload {available_slots} more PDF(s). Currently have {current_doc_count} PDF(s).",
                "current_count": current_doc_count,
                "max_allowed": MAX_PDFS,
                "available_slots": available_slots
            }, status_code=400)
    
    background_tasks.add_task(process_pdfs_background, file_data)
    
    return JSONResponse({
        "status": "started",
        "total": len(file_data),
        "current_count": current_doc_count,
        "max_allowed": MAX_PDFS,
        "message": f"Indexing started. Check progress endpoint for updates. ({current_doc_count + len(file_data)}/{MAX_PDFS} PDFs)"
    })

@app.get("/upload/progress")
async def get_progress():
    progress_data = indexing_progress.copy()
    progress_data["current_pdf_count"] = len(index.docs)
    progress_data["max_pdfs"] = MAX_PDFS
    return JSONResponse(progress_data)

@app.get("/search")
async def search(q: str):
    try:
        if not q or not q.strip():
            return JSONResponse({"results": [], "error": "Empty query"})
        results = index.search(q.strip(), top_k=5)
        return JSONResponse({"results": results})
    except Exception as e:
        print(f"Search endpoint error: {e}")
        return JSONResponse({"results": [], "error": str(e)}, status_code=500)

@app.post("/shutdown")
async def shutdown():
    print("\n Shutdown requested via API endpoint...")
    os.kill(os.getpid(), signal.SIGTERM)
    return JSONResponse({"message": "Server shutdown initiated"})

def signal_handler(sig, frame):
    print("\n\n Shutdown signal received. Stopping server...")
    print("Thank you for using Intelligent PDF Document Retrieval System!")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)