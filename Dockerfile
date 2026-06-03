FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt /app/requirements-deploy.txt
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY deployment /app/deployment
COPY examples /app/examples

ENV PYTHONPATH=/app/deployment

EXPOSE 8080

CMD ["python", "deployment/research_api.py", "--host", "0.0.0.0", "--port", "8080"]
