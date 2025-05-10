
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:1.5b")

# OpenSearch Configuration
OPENSEARCH_URL = os.getenv("OPENSEARCH_URL", "http://opensearch-node:9200")
OPENSEARCH_INDEX = os.getenv("OPENSEARCH_INDEX", "chatbot_faq_index")
