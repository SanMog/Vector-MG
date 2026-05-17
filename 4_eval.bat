@echo off
chcp 65001 >nul
echo ============================================
echo  ВЕКТОР -- Evaluation (retrieval-only, fast)
echo ============================================
echo.
cd /d "%~dp0"

:: Ищем нужный Python (тот что имеет chromadb)
set PYTHON_CMD=
for %%P in (py python python3) do (
    if not defined PYTHON_CMD (
        %%P -c "import chromadb" >nul 2>&1 && set PYTHON_CMD=%%P
    )
)

if not defined PYTHON_CMD (
    echo [!] chromadb не найден. Запустите 1_setup.bat
    pause
    exit /b 1
)

%PYTHON_CMD% evaluate_rag.py --retrieval-only
echo.
pause
