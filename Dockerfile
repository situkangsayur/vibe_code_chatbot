FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -U -r requirements.txt
RUN python -c "import inspect; from langchain_google_genai import embeddings; print(inspect.getmembers(embeddings))"

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]
