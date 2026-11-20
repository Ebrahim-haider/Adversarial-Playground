@echo off
echo ========================================================
echo   Launching Trustworthy AI - Adversarial Playground
echo ========================================================
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py
pause
