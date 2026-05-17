@echo off
chcp 65001 >nul
echo ============================================
echo  ВЕКТОР -- Mass Eval (retrieval-only, fast)
echo ============================================
echo.
cd /d "%~dp0"

py mass_eval.py
if errorlevel 1 python mass_eval.py

echo.
pause
