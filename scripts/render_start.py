"""Restore and validate the release before starting either service."""
import os
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'dvc', 'pull', '-r', 'gdrive_remote',
                'models/random_forest_rul_model.pkl', 'data/processed/train_fd001_processed.csv'], check=True)
from src.agent.release import validate_release
os.environ['APPROVED_MODEL_SHA256'] = validate_release('release/approved.json')
service = sys.argv[1]
if service == 'api':
    command = [sys.executable, '-m', 'uvicorn', 'src.api.main:app', '--host', '0.0.0.0', '--port', os.getenv('PORT', '8000')]
elif service == 'frontend':
    command = [sys.executable, '-m', 'streamlit', 'run', 'app/main.py', '--server.address=0.0.0.0', '--server.port=' + os.getenv('PORT', '8501')]
else:
    raise ValueError('Expected api or frontend')
os.execv(sys.executable, command)
