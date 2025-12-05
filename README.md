# Intelligent PDF Document Retrieval System

## Samarth Patel (CS-410)
It is a web application that performs semantic search across user uploaded PDF documents using a hybrid approach combining BM25 (keyword-based) and BERT (semantic) retrieval algorithms.

## Features

- **Semantic Search**: Goes beyond keyword matching to understand query meaning
- **Hybrid Retrieval**: Combines BM25 and BERT for accurate relevance ranking
- **Multi-Document Search**: Search across multiple PDFs simultaneously
- **Drag & Drop Interface**: Easy PDF upload with modern web UI
- **Local Processing**: All processing happens locally - no cloud dependencies
- **Progress Tracking**: Real-time progress bar for PDF indexing
- **PDF Limit**: Configurable maximum number of PDFs (default: 10)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` to configure settings (default: MAX_PDFS=10)
   - The `.env` file allows you to customize the maximum number of PDFs without modifying code

3. **Note**: The first time you run the application, it will download the BERT model automatically:
   - **Primary**: `all-mpnet-base-v2` (~420MB) - More powerful for semantic understanding
   - **Secondary**: `all-MiniLM-L6-v2` (~80MB) - Faster, lighter alternative
   The system will automatically use the best available model.

## Running the Web Application

1. **Start the server:**

   **Option A: Using the provided scripts (Recommended)**
   
   **For Mac/Linux:**
   ```bash
   # First, make sure the script is executable (one-time setup):
   chmod +x run.sh
   
   # Then run it:
   ./run.sh
   
   # If you get "permission denied", you can also run:
   bash run.sh
   ```
   
   **For Windows:**
   ```bash
   # Simply double-click run.bat or run from command prompt:
   run.bat
   ```
   
   **Option B: Direct command (works on all platforms)**
   ```bash
   uvicorn Project:app --reload
   ```

2. **Open your browser:**
   Navigate to `http://localhost:8000`

3. **Stop the server:**
   - **Method 1 (Recommended)**: Press `Ctrl+C` in the terminal where the server is running
   - **Method 2**: Click the "Shutdown" button in the top-right corner of the web page

4. **Upload PDFs:**
   - Drag and drop PDF files into the upload area, or click to select files
   - Wait for the progress bar to complete indexing

5. **Search:**
   - Enter your query in the search bar
   - Results will show relevant document snippets ranked by relevance score

## How does It Work?

1. **Document Processing**: PDFs are extracted and split into overlapping text chunks
2. **Dual Indexing**: 
   - **BM25**: Creates a keyword-based index for fast retrieval (40% weight)
   - **BERT**: Generates semantic embeddings using sentence transformers (60% weight)
3. **Hybrid Search Process**:
   - **Step 1**: BM25 retrieves top keyword-matching candidates
   - **Step 2**: BERT performs semantic search via FAISS vector similarity
   - **Step 3**: Explicit BERT reranking computes semantic similarity for combined candidates
   - **Step 4**: Weighted fusion combines BM25 and BERT scores (40% BM25 + 60% BERT)
4. **Ranking**: Results are ranked by relevance score, deduplicated (one result per PDF), and displayed with snippets

## Technical Details

- **Backend**: FastAPI (Python)
- **PDF Processing**: PyMuPDF (fitz)
- **BM25 Implementation**: rank-bm25
- **Semantic Embeddings**: sentence-transformers (all-mpnet-base-v2 or all-MiniLM-L6-v2)
- **BERT Implementation**: 
  - Uses transformer-based embeddings for semantic understanding
  - Explicit reranking with cosine similarity computation
  - FAISS vector search for efficient similarity matching
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Frontend**: HTML, CSS, JavaScript (in templates/index.html)
- **Configuration**: Environment variables via .env file

## Requirements

- Python 3.8 or higher
- All dependencies listed in `requirements.txt`

## Project Structure

```
Final Project/
├── Project.py          (Main application - routes, file handling)
├── search.py           (Search engine - Index class, BM25, BERT)
├── templates/
│   └── index.html      (Frontend HTML/CSS/JS)
├── .env                (Environment variables - MAX_PDFS)
├── .env.example        (Template for .env)
├── requirements.txt    (Python dependencies)
├── run.sh              (Startup script for Mac/Linux)
├── run.bat             (Startup script for Windows)
└── README.md           (This file)
```

## Possible Troubleshooting needed: 

### Mac/Linux: "Permission denied" error

If you get `zsh: permission denied: ./run.sh` or `bash: permission denied: ./run.sh`:

**Solution 1: Make the script executable (one-time setup)**
```bash
chmod +x run.sh stop_server.sh
./run.sh
```

**Solution 2: Run with bash directly**
```bash
bash run.sh
```

**Solution 3: Use the direct command**
```bash
uvicorn Project:app --reload --host 0.0.0.0 --port 8000
```

### Windows: Scripts won't run

- Double-click `run.bat` in File Explorer
- Or run from Command Prompt or PowerShell:
  ```cmd
  run.bat
  ```

### Port already in use

If you get an error that port 8000 is already in use:
- Stop any existing server instances
- Or change the port in the scripts/command:
  ```bash
  uvicorn Project:app --reload --host 0.0.0.0 --port 8001
  ```

## Notes

- The index is stored in memory (reason for limited to 10 PDF's) and will be cleared when the server restarts
- For better usercase, adding persistent storage for the index (as perfomed in assignment 1 and 2)
- The system works best with 10+ PDF documents for comprehensive evaluation
- Maximum PDF limit is configurable via `.env` file (default: 10)

