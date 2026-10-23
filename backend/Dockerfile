FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# watchfiles lets --reload skip .venv/ when the source is mounted from the host
RUN pip install --no-cache-dir -r requirements.txt watchfiles

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
