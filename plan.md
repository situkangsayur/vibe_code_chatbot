# Chatbot FAQ Project Plan

## 1. Project Structure:

```
chatbot_faq/
├── data/              # PDF documents
├── app.py             # Streamlit app
├── utils.py           # Utility functions (data loading, chunking, embeddings)
├── config.py          # Configuration settings (API keys, database URL)
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── requirements.txt   # Python dependencies
└── .gitignore         # Ignored files
```

## 2. `requirements.txt`:

```
streamlit
langchain
pymongo
python-dotenv
pdfminer.six
tiktoken
google-generativeai
```

## 3. `docker-compose.yml`:

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - mongodb
    environment:
      MONGODB_URI: mongodb://mongodb:27017/chatbot_faq
      OLLAMA_BASE_URL: http://10.100.21.22:11434 # User provided address

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

## 4. `.gitignore`:

```
.env
data/
__pycache__/
*.pyc
```

## 5. `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI") or "mongodb://localhost:27017/chatbot_faq"
DATABASE_NAME = "chatbot_faq"
COLLECTION_NAME = "embeddings"

# Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434" # Default, but can be overridden by env
OLLAMA_MODEL_NAME = "deepseek-coder" # Or other Deepseek model
```

## 6. `utils.py`:

```python
import os
from typing import List
from pymongo import MongoClient
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GoogleGenerativeAiEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME, GOOGLE_API_KEY

def load_pdf_data(folder_path: str) -> List[str]:
    """Loads and extracts text from all PDF files in the given folder."""
    all_text = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            try:
                text = extract_text(filepath)
                all_text.append(text)
            except Exception as e:
                print(f"Error extracting text from {filename}: {e}")
    return all_text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """Splits the given text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_embeddings(chunks: List[str]) -> GoogleGenerativeAiEmbeddings:
    """Creates embeddings using the Google Generative AI model."""
    return GoogleGenerativeAiEmbeddings(google_api_key=GOOGLE_API_KEY)


def get_mongodb_client():
    """Returns a MongoDB client."""
    return MongoClient(MONGODB_URI)


def store_embeddings_in_mongodb(
    chunks: List[str], embeddings: GoogleGenerativeAiEmbeddings
):
    """Stores the embeddings in MongoDB."""
    client = get_mongodb_client()
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection=collection,
    )
    return vectorstore
```

## 7. `app.py`:

```python
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import GoogleGenerativeAI, Ollama
from langchain.embeddings import GoogleGenerativeAiEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch

from config import (
    GOOGLE_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
    DATABASE_NAME,
    COLLECTION_NAME,
)
from utils import (
    load_pdf_data,
    chunk_text,
    get_mongodb_client,
)


@st.cache_resource
def load_data_and_create_vectorstore():
    """Loads PDF data, chunks it, and creates a vectorstore."""
    st.info("Loading data and creating vectorstore...")
    pdf_folder_path = "data"  # Assuming data folder is in the same directory
    pdf_texts = load_pdf_data(pdf_folder_path)
    all_text = "\n".join(pdf_texts)
    text_chunks = chunk_text(all_text)

    # Use Google Generative AI embeddings
    embeddings = GoogleGenerativeAiEmbeddings(google_api_key=GOOGLE_API_KEY)

    client = get_mongodb_client()
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]

    vectorstore = MongoDBAtlasVectorSearch.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        collection=collection,
    )
    st.success("Data loaded and vectorstore created!")
    return vectorstore


def create_qa_chain(vectorstore, model_choice: str = "gemini"):
    """Creates a question-answering chain."""
    if model_choice == "gemini":
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    elif model_choice == "ollama":
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME)
    else:
        raise ValueError("Invalid model choice. Choose 'gemini' or 'ollama'.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Adjust as needed (stuff, map_reduce, etc.)
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )
    return qa_chain


def main():
    st.title("Chatbot FAQ")

    # Load data and create vectorstore
    vectorstore = load_data_and_create_vectorstore()

    # Model selection
    model_choice = st.selectbox(
        "Choose a language model:", ("gemini", "ollama")
    )

    # Initialize QA chain
    qa_chain = create_qa_chain(vectorstore, model_choice)

    # User input
    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain(query)
            st.write("Answer:", result["result"])

            # Display source documents (optional)
            with st.expander("Source Documents"):
                for doc in result["source_documents"]:
                    st.write(doc)


if __name__ == "__main__":
    main()
```

## 8. `Dockerfile`:

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]