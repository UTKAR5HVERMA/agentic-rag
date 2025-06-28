# app.py
import uvicorn
import uuid
import os
from dotenv import load_dotenv
import uvicorn
import os
import csv
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from feedback import submit_corrected_feedback, process_uploaded_csv, save_feedback_txt
import config
import uuid
# Load environment variables
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from agent import create_agent
from feedback import save_feedback_txt
import config

# --- FastAPI App Setup ---
app = FastAPI(
    title="Bignalytics RAG Agent API (LlamaIndex Version)",
    description="Endpoint for your Bignalytics RAG agent, now powered by LlamaIndex.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Initialization ---
print("🚀 FastAPI Starting - Initializing Bignalytics LlamaIndex Agent...")
print(f"   Multi-Query Mode: {config.ENABLE_MULTI_QUERY}")
# Create and load the agent when the app starts
agent = create_agent()
conversation_history = {} # Simple dictionary to store chat history per conversation
if config.ENABLE_MULTI_QUERY:
    print("✅ Multi-Query Engine ready to serve requests!")
else:
    print("✅ Single-Route Engine ready to serve requests!")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    question: str
    response: str
    feedback: str
    context: Optional[str] = None
    conversation_id: Optional[str] = None
class CorrectedFeedbackRequest(BaseModel):
    question: str
    response: str
    feedback: str
    corrected_response: str
    context: Optional[str] = None
    conversation_id: Optional[str] = None

# --- Helper Functions ---
def format_response(response_text: str) -> str:
    """
    Format the response to ensure proper paragraph breaks and structure.
    """
    # Split by double line breaks to preserve paragraph structure
    paragraphs = response_text.split('\n\n')
    
    # Clean up each paragraph and remove extra whitespace
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned:  # Only add non-empty paragraphs
            cleaned_paragraphs.append(cleaned)
    
    # Join paragraphs with double line breaks for proper formatting
    formatted_response = '\n\n'.join(cleaned_paragraphs)
    
    return formatted_response

# --- API Endpoints ---
@app.post("/ask")
def chat_with_agent(request: ChatRequest):
    """
    Simplified chat handler - the intelligent router handles all intents including greetings.
    """
    convo_id = request.conversation_id or str(uuid.uuid4())

    print(f"--- 📚 Invoking Intelligent Router for: '{request.question}' ---")
    
    # Single router call handles all intents (greetings, bignalytics info, general knowledge)
    response = agent.query(request.question)

    # Format the response for better readability
    formatted_answer = format_response(str(response))

    # Extracting context from source nodes if available
    context_used = ""
    if hasattr(response, 'source_nodes') and response.source_nodes:
        context_used = "\n\n---\n\n".join(
            [node.get_content() for node in response.source_nodes]
        )
    elif hasattr(response, 'metadata') and response.metadata:
        # Extract which tool was selected by the router
        selector_result = response.metadata.get('selector_result', {})
        if hasattr(selector_result, 'selections') and selector_result.selections:
            context_used = f"Router selected: {selector_result.selections[0].reason}"
        else:
            context_used = "Router decision made"

    # Add LLM provider info to context
    llm_provider = "Gemini" if config.USE_GEMINI and config.GEMINI_API_KEY else "Ollama"
    context_used += f"\n\nLLM Provider: {llm_provider}"

    return {
        "answer": formatted_answer,
        "question": request.question,  # Include original question for feedback
        "conversation_id": convo_id,
        "context": context_used
    }

@app.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    """Endpoint to submit feedback."""
    try:
        # Use the actual context from the request, or fall back to a default message
        context_to_save = request.context if request.context else "No context provided"
        
        save_feedback_txt(
            question=request.question,
            context=context_to_save,
            response=request.response,
            feedback=request.feedback,
            file_path=config.FEEDBACK_FILE
        )
        return {"status": "success", "message": "Feedback saved successfully."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save feedback: {str(e)}"}


@app.post("/corrected_feedback")
def submit_corrected_feedback_endpoint(request: CorrectedFeedbackRequest):
    global agent
    try:
        context_to_save = request.context if request.context else "No context provided"
        
        submit_corrected_feedback(
            question=request.question,
            context=context_to_save,
            response=request.response,
            feedback=request.feedback,
            corrected_response=request.corrected_response,
            file_path=config.FEEDBACK_FILE
        )
        agent = create_agent()
        return {"status": "success", "message": "Corrected feedback saved and knowledge base updated."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to save corrected feedback: {str(e)}"}
    
@app.get("/download-feedback_logs")
def download_feedback_logs():
    """Endpoint to download the feedback_logs.csv file."""
    feedback_file_path = config.FEEDBACK_FILE  # Use config.FEEDBACK_FILE instead of hardcoded path
    if os.path.exists(feedback_file_path):
        return FileResponse(
            path=feedback_file_path,
            filename="feedback_logs.csv",
            media_type="text/csv"
        )
    else:
        return {"status": "error", "message": f"File {feedback_file_path} not found."}

@app.get("/read_feedback_logs")
def read_feedback_logs():
    """Endpoint to read the contents of feedback_logs.csv."""
    feedback_file_path = config.FEEDBACK_FILE
    if not os.path.exists(feedback_file_path):
        return {"status": "error", "message": f"File {feedback_file_path} not found.", "data": []}
    
    feedback_data = []
    try:
        with open(feedback_file_path, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                feedback_data.append({
                    "Timestamp": row["Timestamp"],
                    "Question": row["Question"],
                    "Context": row["Context"],
                    "Model Response": row["Model Response"],
                    "Feedback": row["Feedback"],
                    "Corrected Response": row["Corrected Response"]
                })
        return {"status": "success", "data": feedback_data}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read feedback logs: {str(e)}", "data": []}

@app.post("/upload_feedback_logs")
async def upload_feedback_logs(file: UploadFile = File(...)):
    """Endpoint to upload and process a modified feedback_logs.csv file."""
    global agent
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV.")
    
    try:
        content = await file.read()
        # decoded_content = content.decode('utf-8').splitlines()
        try:
            decoded_content = content.decode('utf-8').splitlines()
        except UnicodeDecodeError:
            decoded_content = content.decode('ISO-8859-1').splitlines()
        reader = csv.DictReader(decoded_content)
        uploaded_data = [row for row in reader]
        
        process_uploaded_csv(uploaded_data, file_path=config.FEEDBACK_FILE)
        agent = create_agent()
        return {"status": "success", "message": "Feedback logs uploaded and processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded CSV: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "Bignalytics RAG Agent (LlamaIndex Version) is running."}

# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
