
# 🤖 Chatbot FAQ with OpenSearch + Gemini

Project ini adalah implementasi chatbot dokumen PDF menggunakan:
- LangChain
- Google Generative AI (Gemini)
- OpenSearch sebagai vector database

## 🧱 Arsitektur
1. PDF diambil dari folder `data/`
2. Dipecah jadi chunks, dibuat embedding (Gemini)
3. Disimpan ke OpenSearch
4. Saat user bertanya, sistem ambil dokumen relevan dan berikan jawaban dari LLM

## 🚀 Cara Menjalankan
```bash
docker-compose up --build
```
Akses di browser: http://localhost:8503

## 📦 Stack Teknologi
- LangChain
- Gemini (GoogleGenerativeAI)
- OpenSearch Vector Search
- Streamlit
