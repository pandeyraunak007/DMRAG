"""
Simple Conversational Interface for erwin Data Model
"""
import chromadb
from sentence_transformers import SentenceTransformer

class ErwinChatbot:
    def __init__(self):
        print("🤖 Initializing erwin Data Model Assistant...")
        
        # Set up ChromaDB
        self.client = chromadb.PersistentClient(path="./improved_chroma_db")
        self.collection = self.client.get_collection("erwin-entities")
        
        # Set up embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Assistant ready! Ask me about your data model.")
    
    def search_entities(self, question, top_k=2):
        """Search for relevant entities based on question"""
        # Create query embedding
        query_embedding = self.embedding_model.encode(question)
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"]
        )
        
        return results
    
    def answer_question(self, question):
        """Generate an answer based on the question"""
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching data model...")
        
        # Search for relevant entities
        results = self.search_entities(question)
        
        if not results['ids'] or not results['ids'][0]:
            return "❌ I couldn't find relevant information in the data model."
        
        # Get the best match
        best_match = results['metadatas'][0][0]
        best_score = 1 - results['distances'][0][0]
        best_content = results['documents'][0][0]
        
        # Generate response based on question type
        entity_name = best_match['entity_name']
        subject_area = best_match['subject_area']
        
        if best_score < -0.5:
            return f"❓ I found '{entity_name}' but it doesn't seem very relevant to your question. Could you rephrase?"
        
        # Create a natural response
        response = f"📊 Based on your question, I found the **{entity_name}** entity in the {subject_area} area.\n\n"
        
        if "attribute" in question.lower() or "field" in question.lower():
            response += self._extract_attributes(best_content)
        elif "rule" in question.lower() or "validation" in question.lower():
            response += self._extract_business_rules(best_content)
        elif "description" in question.lower() or "what is" in question.lower():
            response += self._extract_description(best_content)
        else:
            # General overview
            response += self._extract_overview(best_content)
        
        response += f"\n💡 Relevance Score: {best_score:.3f}"
        return response
    
    def _extract_attributes(self, content):
        """Extract attributes from entity content"""
        lines = content.split('\n')
        in_attributes = False
        attributes = []
        
        for line in lines:
            if "Attributes:" in line:
                in_attributes = True
                continue
            elif "Business Rules:" in line:
                in_attributes = False
            elif in_attributes and line.strip():
                attributes.append(line.strip())
        
        if attributes:
            return "🏗️ **Key Attributes:**\n" + "\n".join([f"• {attr}" for attr in attributes[:5]])
        return "No attributes found."
    
    def _extract_business_rules(self, content):
        """Extract business rules from entity content"""
        lines = content.split('\n')
        in_rules = False
        rules = []
        
        for line in lines:
            if "Business Rules:" in line:
                in_rules = True
                continue
            elif in_rules and line.strip():
                rules.append(line.strip())
        
        if rules:
            return "📋 **Business Rules:**\n" + "\n".join([f"• {rule}" for rule in rules[:5]])
        return "No business rules found."
    
    def _extract_description(self, content):
        """Extract description from entity content"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if "Description:" in line and i + 1 < len(lines):
                desc_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith("Attributes:"):
                        desc_lines.append(lines[j].strip())
                    else:
                        break
                if desc_lines:
                    return "📝 **Description:**\n" + " ".join(desc_lines)
        return "No description available."
    
    def _extract_overview(self, content):
        """Extract general overview"""
        desc = self._extract_description(content)
        attrs = self._extract_attributes(content)
        return f"{desc}\n\n{attrs}"
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("🤖 erwin Data Model Assistant")
        print("="*60)
        print("Ask me about your data model! Examples:")
        print("• What is the Customer entity?")
        print("• What attributes does Order have?")
        print("• What are the business rules for Product?")
        print("• Tell me about customer information")
        print("\nType 'quit' to exit, 'help' for more examples")
        print("="*60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using the erwin Assistant!")
                    break
                elif question.lower() == 'help':
                    self.show_help()
                    continue
                elif not question:
                    print("Please ask a question about the data model.")
                    continue
                
                # Generate and display answer
                answer = self.answer_question(question)
                print(f"\n🤖 Assistant: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Sorry, I encountered an error: {e}")
    
    def show_help(self):
        """Show help examples"""
        examples = [
            "What is the Customer entity?",
            "What attributes does the Order entity have?",
            "What are the business rules for Product?",
            "Tell me about customer data",
            "How is order information stored?",
            "What fields are in the product catalog?",
            "What validation rules exist for customers?",
            "Show me product pricing information"
        ]
        
        print("\n💡 Example questions you can ask:")
        for example in examples:
            print(f"  • {example}")

if __name__ == "__main__":
    # Create and start the chatbot
    chatbot = ErwinChatbot()
    chatbot.chat_loop()