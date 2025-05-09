
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

from config import GOOGLE_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, OPENSEARCH_URL, OPENSEARCH_INDEX
from utils import load_pdf_data, chunk_text, create_vectorstore

@st.cache_resource
def load_data_and_create_vectorstore():
    st.info("Loading and embedding PDF documents...")
    pdf_texts = load_pdf_data("data")
    chunks = chunk_text("\n".join(pdf_texts))
    vectorstore = create_vectorstore(chunks)
    st.success("Vectorstore created successfully!")
    return vectorstore

def create_qa_chain(vectorstore, model_choice: str = "gemini"):
    if model_choice == "gemini":
        llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    elif model_choice == "ollama":
        llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME)
    else:
        raise ValueError("Invalid model")
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
    )

def main():
    st.title("Chatbot FAQ (OpenSearch Version)")
    vectorstore = load_data_and_create_vectorstore()
    model_choice = st.selectbox("Choose a model:", ("gemini", "ollama"))
    qa_chain = create_qa_chain(vectorstore, model_choice)
    query = st.text_input("Menanyakan sebuah pertanyaan : ")
    prompt = f"Jawablah pertanyaan ini dalam bahasa Indonesia: {query}"
    if query:
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": prompt})
            st.write("Answer:", result["result"])
            with st.expander("Source Documents"):
                for doc in result["source_documents"]:
                    st.write(doc)

if __name__ == "__main__":
    main()
