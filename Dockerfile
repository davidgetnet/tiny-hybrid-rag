FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install --no-cache-dir \
    grpcio==1.83.0

RUN python -m pip install --no-cache-dir \
    -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["python", "src/inspect_vector_retrieval.py"]