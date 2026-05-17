@echo off
echo ============================================================
echo  Mango Office RAG - Setup
echo ============================================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed
    pause
    exit /b 1
)
echo   OK

echo.
echo [2/3] Pulling embedding model (nomic-embed-text, ~274MB)...
ollama pull nomic-embed-text
if errorlevel 1 (
    echo ERROR: Make sure Ollama is running (check system tray)
    pause
    exit /b 1
)
echo   OK

echo.
echo [3/3] Pulling LLM model (qwen2.5:7b, ~4.7GB)...
echo   This may take a while on first run.
ollama pull qwen2.5:7b
if errorlevel 1 (
    echo ERROR: Failed to pull LLM model
    pause
    exit /b 1
)
echo   OK

echo.
echo ============================================================
echo  Setup complete! Run 2_index.bat next.
echo ============================================================
pause
