
import os
from typing import List
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.schema import Document

from config import GOOGLE_API_KEY, OPENSEARCH_URL, OPENSEARCH_INDEX

def load_pdf_data(folder_path: str) -> List[str]:
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vectorstore(chunks: List[str]) -> OpenSearchVectorSearch:
    documents = [Document(page_content=chunk) for chunk in chunks]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vectorstore = OpenSearchVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        opensearch_url=OPENSEARCH_URL,
        index_name=OPENSEARCH_INDEX,
        bulk_size=1000 
    )
    return vectorstore
