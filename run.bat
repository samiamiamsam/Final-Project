@echo off
echo ==========================================
echo   Intelligent PDF Document Retrieval System
echo ==========================================
echo.
echo Make sure you have installed all dependencies:
echo   pip install -r requirements.txt
echo.
echo Starting server...
echo.
echo Server URL: http://localhost:8000
echo.
echo To stop the server:
echo   - Press Ctrl+C in this window
echo   - Or visit: http://localhost:8000/shutdown (POST request)
echo   - Or run: curl -X POST http://localhost:8000/shutdown
echo.
echo ==========================================
echo.

uvicorn Project:app --reload --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause

