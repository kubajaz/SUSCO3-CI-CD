FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY matrix_benchmark.py .

ENTRYPOINT ["python", "matrix_benchmark.py"]
