import csv
import os
from datetime import datetime
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import config

def save_feedback_txt(question, context, response, feedback, corrected_response=None, file_path="feedback_logs.csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    fieldnames = [
        "Timestamp", "Question", "Context", "Model Response", "Feedback", "Corrected Response"
    ]
    
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                "Timestamp": timestamp,
                "Question": question,
                "Context": context.strip(),
                "Model Response": response.strip(),
                "Feedback": feedback.strip(),
                "Corrected Response": corrected_response.strip() if corrected_response else ""
            }
            
            writer.writerow(row)
        
        print("✅ Feedback saved to CSV")

        if corrected_response:
            # Append to knowledge_base.txt
            with open(config.KNOWLEDGE_BASE_FILE, mode="a", encoding="utf-8") as kb_file:
                kb_entry = f"Q: {question}\nA: {corrected_response}\n"
                kb_file.write(kb_entry)
            print(f"✅ Question and corrected response appended to {config.KNOWLEDGE_BASE_FILE}")

            # PROPERLY UPDATE VECTOR STORE
            embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
            document = Document(text=kb_entry)
            
            # Load existing storage context
            storage_context = StorageContext.from_defaults(persist_dir=config.PERSIST_DIR)
            
            index = load_index_from_storage(storage_context, index_id=config.INDEX_ID, embed_model=embed_model)
            # Insert new document into existing index
            index.insert(document)
            
            # Persist the updated index
            index.storage_context.persist(persist_dir=config.PERSIST_DIR)
            print("✅ Vector store updated with new embedding")
        
    except Exception as e:
        print(f"❌ Failed to save feedback or update knowledge base: {str(e)}")
        raise
        
def submit_corrected_feedback(question, context, response, feedback, corrected_response, file_path="feedback_logs.csv"):
    save_feedback_txt(
        question=question,
        context=context,
        response=response,
        feedback=feedback,
        corrected_response=corrected_response,
        file_path=file_path
    )


def process_uploaded_csv(uploaded_data, file_path="feedback_logs.csv"):
    fieldnames = ["Timestamp", "Question", "Context", "Model Response", "Feedback", "Corrected Response"]
    
    try:
        # Load existing index
        embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
        storage_context = StorageContext.from_defaults(persist_dir=config.PERSIST_DIR)
        
        # Check if index exists; if not, create a new one
        try:
            index = load_index_from_storage(
                storage_context, 
                index_id=config.INDEX_ID,  # Specify index_id
                embed_model=embed_model
            )
            print("   ✅ Loaded existing index from storage.")
        except Exception as e:
            print(f"   ⚠️ No existing index found: {str(e)}. Creating new index...")
            index = VectorStoreIndex.from_documents(
                [], 
                storage_context=storage_context, 
                embed_model=embed_model,
                index_id=config.INDEX_ID  # Specify index_id
            )
            print("   ✅ New index created.")
        
        existing_data = {}
        if os.path.exists(file_path):
            with open(file_path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    key = f"{row['Question']}_{row['Timestamp']}"
                    existing_data[key] = row

        # Write uploaded data to CSV
        with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in uploaded_data:
                writer.writerow({
                    "Timestamp": row.get("Timestamp", ""),
                    "Question": row.get("Question", ""),
                    "Context": row.get("Context", ""),
                    "Model Response": row.get("Model Response", ""),
                    "Feedback": row.get("Feedback", ""),
                    "Corrected Response": row.get("Corrected Response", "")
                })

        # Process corrections and update vector store
        for row in uploaded_data:
            key = f"{row['Question']}_{row['Timestamp']}"
            corrected_response = row.get("Corrected Response", "")
            if corrected_response:
                existing_row = existing_data.get(key, {})
                existing_corrected = existing_row.get("Corrected Response", "")
                if corrected_response != existing_corrected:
                    with open(config.KNOWLEDGE_BASE_FILE, mode="a", encoding="utf-8") as kb_file:
                        kb_entry = f"Q: {row['Question']}\nA: {corrected_response}\n"
                        kb_file.write(kb_entry)
                        print(f"✅ Appended to {config.KNOWLEDGE_BASE_FILE}: {kb_entry.strip()}")

                    # Insert new document into existing index
                    document = Document(text=kb_entry)
                    index.insert(document)
                    print("✅ Document inserted into vector store")

        # Persist the updated index
        index.storage_context.persist(persist_dir=config.PERSIST_DIR)
        # from agent import refresh_vector_index
        # refresh_vector_index()  # Ensure the vector index is refreshed after updates
        print("✅ Vector store updated with new embeddings")
        print("✅ Uploaded CSV processed and feedback_logs.csv updated")
        
    except Exception as e:
        print(f"❌ Failed to process uploaded CSV: {str(e)}")
        raise