# config_li.py
import os

# --- File Paths ---
KNOWLEDGE_BASE_FILE = "knowledge_base.txt"
PERSIST_DIR = "./llama_index_storage" # Directory for LlamaIndex vector store
FEEDBACK_FILE = "feedback_logs.csv"
INDEX_ID = "bignalytics_vector_index"
# --- LLM Provider Selection ---
USE_OLLAMA = False  # Set to True to use Ollama, False to use Gemini
USE_GEMINI = True  # Set to True to use Gemini, False to use Ollama

# --- Ollama Configuration ---
LLM_MODEL = "gemma3:latest" # Model to run on Ollama

# --- Gemini Configuration ---
GEMINI_MODEL = "gemma-3-12b-it"  # Gemini model to use (gemini-1.5-flash supports Gemma)
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')  # Get from environment variable

# --- Embedding and General Settings ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
REQUEST_TIMEOUT = 120.0 # Timeout for LLM request

# --- RAG Settings ---
RERANK_TOP_N =2  # Number of results needed after reranking (reduced for token efficiency)
SIMILARITY_TOP_K = 5# Number of results to fetch initially in vector search (reduced for token efficiency)

# --- Chunking Settings (for token optimization) ---
CHUNK_SIZE = 512 # Reduce from default 1024 to save tokens
CHUNK_OVERLAP = 50 # Reduce from default 200 to save tokens

# --- Multi-Query Engine Settings ---
ENABLE_MULTI_QUERY = True  # Enable multi-query decomposition
MAX_SUB_QUERIES = 10  # Maximum number of sub-queries to generate
MULTI_QUERY_VERBOSE = True  # Show multi-query processing details
