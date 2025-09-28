"""
Web Chat Interface for erwin RAG System
"""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import chromadb
from sentence_transformers import SentenceTransformer
import json
import uvicorn
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="erwin RAG Chat Interface", description="Conversational AI for erwin Data Models")

# Set up templates (we'll create this)
templates = Jinja2Templates(directory="templates")

class ErwinWebChatbot:
    def __init__(self):
        print("🤖 Initializing Web Chatbot...")
        
        # Set up ChromaDB with enterprise data
        self.client = chromadb.PersistentClient(path="./enterprise_chroma_db")
        self.collection = self.client.get_collection("enterprise-erwin-model")
        
        # Set up embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Web Chatbot ready!")
    
    def search_and_respond(self, question):
        """Search and generate response for web interface"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode(question)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3,
                include=["metadatas", "documents", "distances"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                return {
                    "success": False,
                    "response": "❌ I couldn't find relevant information in the data model.",
                    "sources": [],
                    "confidence": 0
                }
            
            # Get best match
            best_score = 1 - results['distances'][0][0]
            best_metadata = results['metadatas'][0][0]
            best_content = results['documents'][0][0]
            
            # Generate response
            response_text = self._generate_web_response(question, best_metadata, best_content, best_score)
            
            # Collect sources
            sources = []
            for i in range(min(3, len(results['ids'][0]))):
                sources.append({
                    "name": results['metadatas'][0][i].get('entity_name', 
                                 results['metadatas'][0][i].get('relationship_name',
                                 results['metadatas'][0][i].get('subject_area', 'Unknown'))),
                    "type": results['metadatas'][0][i]['type'],
                    "score": 1 - results['distances'][0][i],
                    "subject_area": results['metadatas'][0][i].get('subject_area', 'N/A')
                })
            
            return {
                "success": True,
                "response": response_text,
                "sources": sources,
                "confidence": best_score,
                "query": question
            }
            
        except Exception as e:
            return {
                "success": False,
                "response": f"❌ Error processing your question: {str(e)}",
                "sources": [],
                "confidence": 0
            }
    
    def _generate_web_response(self, question, metadata, content, score):
        """Generate web-friendly response"""
        content_type = metadata['type']
        
        if content_type == 'entity':
            return self._format_entity_response(metadata, content, score)
        elif content_type == 'relationship':
            return self._format_relationship_response(metadata, content, score)
        elif content_type == 'subject_area':
            return self._format_subject_area_response(metadata, content, score)
        else:
            return f"Found information about {metadata.get('entity_name', 'the data model')}. Confidence: {score:.3f}"
    
    def _format_entity_response(self, metadata, content, score):
        """Format entity response for web"""
        entity_name = metadata['entity_name']
        subject_area = metadata['subject_area']
        attr_count = metadata.get('attribute_count', 0)
        rule_count = metadata.get('rule_count', 0)
        
        response = f"**{entity_name}** is an entity in the {subject_area} subject area.\n\n"
        
        # Extract description
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "Description:" in line and i + 1 < len(lines):
                desc_start = i + 1
                desc_lines = []
                for j in range(desc_start, len(lines)):
                    if lines[j].strip() and not lines[j].startswith("Attributes:"):
                        desc_lines.append(lines[j].strip())
                    else:
                        break
                if desc_lines:
                    response += " ".join(desc_lines) + "\n\n"
                break
        
        response += f"📊 **Key Details:**\n"
        response += f"• Contains {attr_count} attributes\n"
        response += f"• Has {rule_count} business rules\n"
        response += f"• Part of {subject_area}\n\n"
        
        # Add a few key attributes
        in_attributes = False
        attributes = []
        for line in lines:
            if "Attributes" in line and ":" in line:
                in_attributes = True
                continue
            elif "Business Rules" in line:
                break
            elif in_attributes and line.strip().startswith('•'):
                attributes.append(line.strip()[2:])
                if len(attributes) >= 3:
                    break
        
        if attributes:
            response += "🔧 **Sample Attributes:**\n"
            for attr in attributes:
                response += f"• {attr}\n"
        
        return response
    
    def _format_relationship_response(self, metadata, content, score):
        """Format relationship response for web"""
        rel_name = metadata['relationship_name']
        parent = metadata['parent_entity']
        child = metadata['child_entity']
        rel_type = metadata['relationship_type']
        
        response = f"**{rel_name}** is a {rel_type} relationship.\n\n"
        response += f"🔗 **Connection:** {parent} → {child}\n\n"
        
        # Extract description
        lines = content.split('\n')
        for line in lines:
            if "Description:" in line:
                idx = lines.index(line)
                if idx + 1 < len(lines):
                    response += lines[idx + 1].strip()
                break
        
        return response
    
    def _format_subject_area_response(self, metadata, content, score):
        """Format subject area response for web"""
        subject_area = metadata['subject_area']
        entity_count = metadata.get('entity_count', 0)
        
        response = f"**{subject_area}** is a subject area containing {entity_count} entities.\n\n"
        
        # Extract entities
        lines = content.split('\n')
        in_entities = False
        entities = []
        
        for line in lines:
            if "Entities in this Subject Area:" in line:
                in_entities = True
                continue
            elif in_entities and line.strip().startswith('•'):
                entities.append(line.strip()[2:])
            elif in_entities and line.strip() and not line.strip().startswith('•'):
                break
        
        if entities:
            response += "📋 **Contains these entities:**\n"
            for entity in entities:
                response += f"• {entity}\n"
        
        return response

# Initialize chatbot
chatbot = ErwinWebChatbot()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    # Send welcome message
    welcome_message = {
        "type": "assistant",
        "message": "👋 Hello! I'm your erwin Data Model Assistant. Ask me about entities, attributes, business rules, or relationships in your data model!",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "sources": []
    }
    await websocket.send_text(json.dumps(welcome_message))
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "").strip()
            
            if not user_message:
                continue
            
            # Process the question
            response = chatbot.search_and_respond(user_message)
            
            # Send response back to client
            response_message = {
                "type": "assistant",
                "message": response["response"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "sources": response["sources"],
                "confidence": response["confidence"],
                "success": response["success"]
            }
            
            await websocket.send_text(json.dumps(response_message))
            
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    print("🚀 Starting erwin RAG Web Chat Interface...")
    print("📱 Open your browser to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)