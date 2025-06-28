#default systesizer

# agent.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    SummaryIndex,
    Document,
    set_global_handler,
)
from llama_index.core import get_response_synthesizer, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.question_gen import LLMQuestionGenerator
from gemini_key_manager import GeminiKeyManager

# Import configuration and prompts
import config
from prompts import get_professional_greeting_prompt
# In this new architecture, the agent won't need a direct system prompt,
# as the router will make the decisions.

# Initialize GeminiKeyManager globally
key_manager = GeminiKeyManager("gemini_keys.txt")

def initialize_global_settings():
    """Initializes the LLM and embedding model globally, with Gemini key rotation."""
    print("1. Initializing Global LLM and Embedding Model Settings...")
    
    # Enable LangSmith tracing via environment variables (minimal setup)
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "biganalytics-agent")
        print("🔍 LangSmith tracing enabled via environment variables")
    
    # Select LLM provider based on configuration
    if config.USE_GEMINI:
        for _ in range(len(key_manager.keys)):
            gemini_api_key = key_manager.get_key()
            try:
                Settings.llm = Gemini(
                    model=config.GEMINI_MODEL,
                    api_key=gemini_api_key,
                    temperature=0.1
                )
                print(f"✅ Gemini LLM ({config.GEMINI_MODEL}) initialized with key index {key_manager.index}.")
                break
            except Exception as e:
                print(f"❌ Gemini key at index {key_manager.index} failed: {e}")
                key_manager.rotate_key()
        else:
            raise RuntimeError("All Gemini API keys failed.")
    elif config.USE_OLLAMA:
        print(f"Using Ollama LLM: {config.LLM_MODEL}")
        Settings.llm = Ollama(model=config.LLM_MODEL, request_timeout=config.REQUEST_TIMEOUT)
        print(f"✅ Ollama LLM ({config.LLM_MODEL}) initialized.")
    else:
        raise ValueError("No valid LLM provider configured. Please set either USE_OLLAMA=True or USE_GEMINI=True with GEMINI_API_KEY.")
    
    Settings.embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    print(f"✅ Embed Model ({config.EMBED_MODEL}) globally set.")
    
def get_vector_index():
    """Loads or creates the knowledge base vector index with optimized chunking."""
    print(f"2. Loading or Creating Vector Store Index from '{config.PERSIST_DIR}'...")
    
    # Check if storage directory exists and contains required files
    required_files = ['docstore.json', 'index_store.json', 'vector_store.json']
    storage_dir_exists = os.path.exists(config.PERSIST_DIR)
    all_files_exist = storage_dir_exists and all(os.path.exists(os.path.join(config.PERSIST_DIR, f)) for f in required_files)
    
    if not storage_dir_exists or not all_files_exist:
        print(f"   Storage directory missing or incomplete. Creating new index from '{config.KNOWLEDGE_BASE_FILE}'...")
        # Ensure knowledge_base.txt exists
        if not os.path.exists(config.KNOWLEDGE_BASE_FILE):
            print(f"   {config.KNOWLEDGE_BASE_FILE} not found. Creating empty file...")
            with open(config.KNOWLEDGE_BASE_FILE, 'a', encoding='utf-8') as f:
                pass  # Create empty file if it doesn't exist
        
        # Initialize storage context
        storage_context = StorageContext.from_defaults()
        
        documents = SimpleDirectoryReader(input_files=[config.KNOWLEDGE_BASE_FILE]).load_data()

        # Configure optimized chunking settings to reduce token usage
        print(f"   Using optimized chunking: {config.CHUNK_SIZE} tokens with {config.CHUNK_OVERLAP} overlap...")
        text_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,  # Reduced from default 1024
            chunk_overlap=config.CHUNK_OVERLAP,  # Reduced from default 200
            separator=" "
        )
        
        index = VectorStoreIndex.from_documents(
            documents, 
            transformations=[text_splitter],
            storage_context=storage_context,
            index_id=config.INDEX_ID  # Specify index_id
        )
        index.storage_context.persist(persist_dir=config.PERSIST_DIR)
        print("   ✅ New index created and persisted with optimized chunking.")
    else:
        print("   Found existing storage. Loading index...")
        storage_context = StorageContext.from_defaults(persist_dir=config.PERSIST_DIR)
        index = load_index_from_storage(storage_context, index_id=config.INDEX_ID)
        print("   ✅ Index loaded from storage.")
    return index

def refresh_vector_index():
    """Refreshes the vector index with updated knowledge base"""
    print("Refreshing vector index with updated knowledge...")
    try:
        documents = SimpleDirectoryReader(input_files=[config.KNOWLEDGE_BASE_FILE]).load_data()
        text_splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separator=" "
        )
        storage_context = StorageContext.from_defaults(persist_dir=config.PERSIST_DIR)
        index = VectorStoreIndex.from_documents(
            documents, 
            transformations=[text_splitter],
            storage_context=storage_context,
            index_id=config.INDEX_ID  # Specify index_id
        )
        index.storage_context.persist(persist_dir=config.PERSIST_DIR)
        print("✅ Vector index refreshed with updated knowledge")
    except Exception as e:
        print(f"❌ Failed to refresh vector index: {str(e)}")
        raise


def create_multi_query_engine():
    """
    Creates a multi-query engine that can decompose complex queries into sub-questions
    and route them to appropriate experts for comprehensive answers.
    """
    initialize_global_settings()
    index = get_vector_index()

    print("3. Creating Expert Query Engines for Multi-Query System...")

    # --- Expert 1: Bignalytics Knowledge Expert (RAG System) ---
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-12-v2", top_n=config.RERANK_TOP_N
    )
    bignalytics_query_engine = index.as_query_engine(
        similarity_top_k=config.SIMILARITY_TOP_K,
        node_postprocessors=[reranker],
        response_mode="compact",
    )
    print("   ✅ Bignalytics RAG expert ready.")

    # --- Expert 2: General Knowledge Expert (Direct LLM) ---
    general_doc = Document(text="You are a helpful AI assistant. Answer questions about programming, data science, and general knowledge topics accurately and concisely.")
    general_index = SummaryIndex.from_documents([general_doc])
    general_query_engine = general_index.as_query_engine(
        response_mode="compact"
    )
    print("   ✅ General Knowledge LLM expert ready.")

    # --- Expert 3: Greeting & Conversation Expert (Professional) ---
    greeting_doc = Document(text=get_professional_greeting_prompt())
    greeting_index = SummaryIndex.from_documents([greeting_doc])
    greeting_query_engine = greeting_index.as_query_engine(
        response_mode="compact"
    )
    print("   ✅ Greeting & Conversation expert ready.")

    print("4. Creating Query Engine Tools...")
    # Create tools for multi-query engine
    bignalytics_tool = QueryEngineTool(
        query_engine=bignalytics_query_engine,
        metadata=ToolMetadata(
            name="bignalytics_rag_tool",
            description=(
                "Provides specific information about Bignalytics Training Institute including: "
                "courses offered (Data Science, Data Analytics, AI/ML), course fees and duration, "
                "placement assistance, faculty information, admission process, contact details, "
                "and any other institute-specific information."
            ),
        ),
    )

    general_tool = QueryEngineTool(
        query_engine=general_query_engine,
        metadata=ToolMetadata(
            name="general_knowledge_tool",
            description=(
                "Answers general questions about programming languages (Python, Java, etc.), "
                "data science concepts, machine learning algorithms, statistics, mathematics, "
                "technology concepts, and other general knowledge not specific to Bignalytics institute."
            ),
        ),
    )

    greeting_tool = QueryEngineTool(
        query_engine=greeting_query_engine,
        metadata=ToolMetadata(
            name="greeting_conversation_tool",
            description=(
                "Handles social interactions including greetings (hello, hi), "
                "conversational responses, thank you messages, goodbye messages, "
                "and maintains professional tone while providing Bignalytics contact information."
            ),
        ),
    )
    print("   ✅ Query engine tools created.")

    print("5. Creating Multi-Query Engine with Question Generator...")
    
    # Create question generator for decomposing complex queries
    question_gen = LLMQuestionGenerator.from_defaults(
        llm=Settings.llm
    )
    from llama_index.core import get_response_synthesizer

    response_synthesizer = get_response_synthesizer(
        response_mode="context_only"
    )
    # Create the multi-query engine
    
    # Creating custom synthesizer
    custom_summary_prompt = PromptTemplate("""PromptGiven the following context and query, synthesize a detailed and informative summary that directly addresses the query. Ensure the summary includes all key details relevant to the query from the provided context, maintaining clarity and avoiding excessive compression. Do not omit critical information or overly condense the response. Structure the summary to be concise yet comprehensive, organized logically, and focused on answering the query fully.
        Instruction while reponding do not include sub quetiona and response tag just prove a proper response
        ------------
        Context:  
        {context_str}  
        ------------  
        Query:  
        {query_str}  
        ------------  
        Summary:
        """
    )  
    from llama_index.core.response_synthesizers import ResponseMode
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.TREE_SUMMARIZE,
        summary_template=custom_summary_prompt
    )
    multi_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=[
            bignalytics_tool,
            general_tool,
            greeting_tool,
        ],
        response_synthesizer=response_synthesizer,
        question_gen=question_gen,
        verbose=config.MULTI_QUERY_VERBOSE,
        use_async=False  # Set to False for simpler execution
    )
    
    print("✅ Multi-Query Engine successfully created!")
    print(f"   📊 Max sub-queries: {config.MAX_SUB_QUERIES}")
    print(f"   🔍 Verbose mode: {config.MULTI_QUERY_VERBOSE}")
    
    return multi_query_engine

def create_agent():
    """
    Master function that creates either a single-route or multi-query engine
    based on configuration settings.
    """
    return create_multi_query_engine()
    # if config.ENABLE_MULTI_QUERY:
    #     print("🔄 Creating Multi-Query Engine (supports query decomposition)...")
    #     return create_multi_query_engine()
    # else:
    #     print("🎯 Creating Single-Route Engine (traditional routing)...")
    #     return create_intelligent_query_engine()

# --- Main execution block for testing ---
if __name__ == "__main__":
    print("--- Testing Multi-Query Engine vs Single-Route Engine ---")
    
    # Test the configured agent (will be multi-query if ENABLE_MULTI_QUERY=True)
    print(f"\n--- Current Configuration: ENABLE_MULTI_QUERY = {config.ENABLE_MULTI_QUERY} ---")
    agent = create_agent()
    
    # Test 1: Simple query
    print("\n--- Test 1: Simple Institute Query ---")
    response1 = agent.query("What are the fees for the Data Science course?")
    print("\n--- Answer 1 ---")
    print(response1)
    
    # Test 2: Complex hybrid query (perfect for multi-query)
    print("\n--- Test 2: Complex Hybrid Query ---")
    complex_query = "What is machine learning and what ML courses does Bignalytics offer with their fees?"
    response2 = agent.query(complex_query)
    print("\n--- Answer 2 ---")
    print(response2)
    
    # Test 3: Multi-part question
    print("\n--- Test 3: Multi-Part Question ---")
    multipart_query = "Hello! Can you explain data science and tell me about Bignalytics' data science course fees and duration?"
    response3 = agent.query(multipart_query)
    print("\n--- Answer 3 ---")
    print(response3)
    
    print(f"\n--- Testing Complete ---")
    if config.ENABLE_MULTI_QUERY:
        print("✅ Multi-Query Engine: Decomposes complex queries into sub-questions")
        print("   - Handles complex queries by breaking them down")
        print("   - Routes sub-questions to appropriate experts")
        print("   - Combines responses for comprehensive answers")
    else:
        print("✅ Single-Route Engine: Routes entire query to one expert")
        print("   - Fast routing decision")
        print("   - One expert handles the complete query")
        print("   - Good for queries with single intent")
