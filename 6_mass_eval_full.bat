@echo off
chcp 65001 >nul
echo ============================================
echo  VEKTOR -- Mass Eval FULL (LLM generation)
echo ============================================
echo.
echo Mode: full LLM answer per question
echo Model: qwen2.5:14b  (Ollama must be running)
echo Questions: ~98,  estimated time: 45-60 min
echo.
cd /d "%~dp0"

py mass_eval.py --generate
if errorlevel 1 python mass_eval.py --generate

echo.
pause
