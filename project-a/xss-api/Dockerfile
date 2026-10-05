FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY project-a/xss-api/app.py /app/app.py
COPY project-a/xss-api/requirements.txt /app/requirements.txt

COPY models /models

RUN pip install --no-cache-dir -r /app/requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]