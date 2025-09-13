@echo off
REM Activate venv and run training with default data path
if not exist .venv (
    echo Creating virtual environment...
    py -m venv .venv
)
call .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python src\train.py --data data\Telco-Customer-Churn.csv
