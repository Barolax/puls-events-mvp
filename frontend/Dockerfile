FROM python:3.12-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/
COPY agents/ ./agents/
COPY .env .

ENV PYTHONPATH=/app

CMD ["chainlit", "run", "frontend/app.py", "--host", "0.0.0.0", "--port", "8001"]