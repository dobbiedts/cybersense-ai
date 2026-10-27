FROM python:3.11-slim

WORKDIR /app
COPY waf-logger/app.py .

# Standard library only, no pip install needed
CMD ["python", "-u", "app.py"]