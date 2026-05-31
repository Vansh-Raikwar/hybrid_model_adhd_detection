@echo off
set "VENV_PATH=%~dp0venv"

if not exist "%VENV_PATH%" (
    echo Virtual environment not found. Please wait while it's created...
    python -m venv venv
    call "%VENV_PATH%\Scripts\activate"
    pip install -r requirements.txt
) else (
    call "%VENV_PATH%\Scripts\activate"
)

echo Starting ADHD Detection AI...
python -m streamlit run app.py
pause
