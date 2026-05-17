@echo off
echo ============================================================
echo  Mango Office RAG - Starting chat
echo ============================================================
echo.
echo  Opening browser at http://localhost:8501
echo  Press Ctrl+C in this window to stop.
echo.

cd /d "%~dp0"
python -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
