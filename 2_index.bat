@echo off
echo ============================================================
echo  Mango Office RAG - Indexing documents
echo ============================================================
echo.
echo  Reading PDFs from docs\ folder and Jira TXT export, chunking, embedding...
echo  This will take 5-20 minutes depending on document size.
echo.

cd /d "%~dp0"
python indexer.py

if errorlevel 1 (
    echo.
    echo ERROR during indexing. Check output above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Run 3_start.bat to open the chat.
echo ============================================================
pause
