FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN python -m scripts.restore_bundle

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:' + __import__('os').getenv('PORT', '8501') + '/_stcore/health')"

EXPOSE 8501

CMD streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0
