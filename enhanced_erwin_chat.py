"""
Enhanced Conversational Interface for Enterprise erwin Data Model
"""
import chromadb
from sentence_transformers import SentenceTransformer

class EnhancedErwinChatbot:
    def __init__(self):
        print("🤖 Initializing Enhanced erwin Data Model Assistant...")
        
        # Set up ChromaDB with enterprise data
        self.client = chromadb.PersistentClient(path="./enterprise_chroma_db")
        self.collection = self.client.get_collection("enterprise-erwin-model")
        
        # Set up embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Enhanced Assistant ready! Ask me about your enterprise data model.")
    
    def search_entities(self, question, top_k=3):
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
        print("🔍 Searching enterprise data model...")
        
        # Search for relevant information
        results = self.search_entities(question, top_k=3)
        
        if not results['ids'] or not results['ids'][0]:
            return "❌ I couldn't find relevant information in the data model."
        
        # Get the best matches
        matches = []
        for i in range(len(results['ids'][0])):
            score = 1 - results['distances'][0][i]
            metadata = results['metadatas'][0][i]
            content = results['documents'][0][i]
            
            matches.append({
                'score': score,
                'metadata': metadata,
                'content': content
            })
        
        # Generate response based on best match
        best_match = matches[0]
        
        if best_match['score'] < -0.5:
            return f"❓ I found some information but it doesn't seem very relevant. Could you rephrase your question?"
        
        # Create response based on content type
        metadata = best_match['metadata']
        content_type = metadata['type']
        
        if content_type == 'entity':
            return self._generate_entity_response(question, matches)
        elif content_type == 'relationship':
            return self._generate_relationship_response(question, matches)
        elif content_type == 'subject_area':
            return self._generate_subject_area_response(question, matches)
        else:
            return self._generate_general_response(question, matches)
    
    def _generate_entity_response(self, question, matches):
        """Generate response for entity-related questions"""
        best_match = matches[0]
        metadata = best_match['metadata']
        content = best_match['content']
        
        entity_name = metadata['entity_name']
        subject_area = metadata['subject_area']
        attr_count = metadata.get('attribute_count', 0)
        rule_count = metadata.get('rule_count', 0)
        
        response = f"📊 **{entity_name}** Entity\n"
        response += f"📂 Subject Area: {subject_area}\n"
        response += f"📋 Attributes: {attr_count} | Business Rules: {rule_count}\n\n"
        
        # Extract specific information based on question
        if any(word in question.lower() for word in ['attribute', 'field', 'column']):
            response += self._extract_attributes(content)
        elif any(word in question.lower() for word in ['rule', 'validation', 'constraint']):
            response += self._extract_business_rules(content)
        elif any(word in question.lower() for word in ['description', 'what is', 'purpose']):
            response += self._extract_description(content)
        else:
            # General overview
            response += self._extract_description(content)
            response += "\n\n" + self._extract_key_attributes(content, 5)
        
        # Add related entities if any
        related = self._find_related_matches(matches[1:], entity_name)
        if related:
            response += f"\n\n🔗 **Related Information:**\n{related}"
        
        response += f"\n\n💡 Relevance Score: {best_match['score']:.3f}"
        return response
    
    def _generate_relationship_response(self, question, matches):
        """Generate response for relationship questions"""
        best_match = matches[0]
        metadata = best_match['metadata']
        content = best_match['content']
        
        rel_name = metadata['relationship_name']
        parent = metadata['parent_entity']
        child = metadata['child_entity']
        rel_type = metadata['relationship_type']
        
        response = f"🔗 **{rel_name}** Relationship\n"
        response += f"📊 Type: {rel_type}\n"
        response += f"🔄 Connection: {parent} → {child}\n\n"
        response += self._extract_description(content)
        
        response += f"\n\n💡 Relevance Score: {best_match['score']:.3f}"
        return response
    
    def _generate_subject_area_response(self, question, matches):
        """Generate response for subject area questions"""
        best_match = matches[0]
        metadata = best_match['metadata']
        content = best_match['content']
        
        subject_area = metadata['subject_area']
        entity_count = metadata.get('entity_count', 0)
        total_attrs = metadata.get('total_attributes', 0)
        total_rules = metadata.get('total_rules', 0)
        
        response = f"📂 **{subject_area}** Subject Area\n"
        response += f"📊 Contains: {entity_count} entities, {total_attrs} attributes, {total_rules} business rules\n\n"
        response += self._extract_description(content)
        
        # Extract entity list
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
            response += f"\n\n🏗️ **Entities:**\n"
            for entity in entities:
                response += f"• {entity}\n"
        
        response += f"\n\n💡 Relevance Score: {best_match['score']:.3f}"
        return response
    
    def _generate_general_response(self, question, matches):
        """Generate general response"""
        best_match = matches[0]
        metadata = best_match['metadata']
        
        response = f"📋 **Information Found**\n"
        response += f"Type: {metadata['type'].title()}\n\n"
        response += self._extract_description(best_match['content'])
        response += f"\n\n💡 Relevance Score: {best_match['score']:.3f}"
        return response
    
    def _extract_attributes(self, content):
        """Extract attributes from content"""
        lines = content.split('\n')
        in_attributes = False
        attributes = []
        
        for line in lines:
            if "Attributes" in line and ":" in line:
                in_attributes = True
                continue
            elif "Business Rules" in line or "Relationship:" in line:
                in_attributes = False
            elif in_attributes and line.strip().startswith('•'):
                attributes.append(line.strip()[2:])
        
        if attributes:
            return "🏗️ **Key Attributes:**\n" + "\n".join([f"• {attr}" for attr in attributes[:8]])
        return "No attributes found."
    
    def _extract_key_attributes(self, content, limit=5):
        """Extract limited number of key attributes"""
        lines = content.split('\n')
        in_attributes = False
        attributes = []
        
        for line in lines:
            if "Attributes" in line and ":" in line:
                in_attributes = True
                continue
            elif "Business Rules" in line:
                in_attributes = False
            elif in_attributes and line.strip().startswith('•'):
                attributes.append(line.strip()[2:])
                if len(attributes) >= limit:
                    break
        
        if attributes:
            return "🏗️ **Key Attributes:**\n" + "\n".join([f"• {attr}" for attr in attributes])
        return ""
    
    def _extract_business_rules(self, content):
        """Extract business rules from content"""
        lines = content.split('\n')
        in_rules = False
        rules = []
        
        for line in lines:
            if "Business Rules" in line and ":" in line:
                in_rules = True
                continue
            elif in_rules and line.strip().startswith('•'):
                rules.append(line.strip()[2:])
        
        if rules:
            return "📋 **Business Rules:**\n" + "\n".join([f"• {rule}" for rule in rules[:6]])
        return "No business rules found."
    
    def _extract_description(self, content):
        """Extract description from content"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if "Description:" in line:
                desc_lines = []
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not any(keyword in lines[j] for keyword in ['Attributes:', 'Business Rules:', 'Entities in this']):
                        desc_lines.append(lines[j].strip())
                    elif lines[j].strip() and any(keyword in lines[j] for keyword in ['Attributes:', 'Business Rules:']):
                        break
                
                if desc_lines:
                    description = " ".join(desc_lines)
                    return f"📝 **Description:**\n{description}"
        
        return "📝 **Description:** No description available."
    
    def _find_related_matches(self, other_matches, entity_name):
        """Find related information in other matches"""
        related_info = []
        
        for match in other_matches:
            if match['score'] > 0:  # Only include positive matches
                metadata = match['metadata']
                if metadata['type'] == 'relationship':
                    if entity_name in [metadata.get('parent_entity'), metadata.get('child_entity')]:
                        related_info.append(f"• {metadata['relationship_name']} ({metadata['relationship_type']})")
                elif metadata['type'] == 'entity':
                    related_info.append(f"• {metadata['entity_name']} entity ({metadata['subject_area']})")
        
        return "\n".join(related_info[:3]) if related_info else ""
    
    def run_test_queries(self):
        """Run comprehensive test queries on the enterprise data"""
        print("\n" + "="*70)
        print("🧪 COMPREHENSIVE TEST OF ENTERPRISE DATA MODEL")
        print("="*70)
        
        test_categories = {
            "Basic Entity Queries": [
                "What is the Customer entity?",
                "Tell me about the Product entity",
                "What is the Order entity?",
                "Describe the Payment entity"
            ],
            "Attribute Questions": [
                "What attributes does Customer have?",
                "What fields are in the Product entity?",
                "Show me Order attributes",
                "What data is stored in CustomerAddress?"
            ],
            "Business Rules": [
                "What are the business rules for Customer?",
                "What validation rules exist for Product?",
                "Tell me about Order business rules",
                "What constraints exist for Payment?"
            ],
            "Subject Area Questions": [
                "Tell me about Customer Management",
                "What is in the Product Catalog area?",
                "Describe Order Management",
                "What's in Financial Management?"
            ],
            "Relationship Questions": [
                "How are Customer and Order related?",
                "What's the relationship between Product and OrderItem?",
                "How do Customer and CustomerAddress connect?",
                "Tell me about Order and Payment relationships"
            ],
            "Complex Queries": [
                "customer information and personal data",
                "product catalog and inventory management",
                "order processing workflow",
                "payment transaction handling",
                "business rules and validation requirements"
            ]
        }
        
        all_scores = []
        
        for category, queries in test_categories.items():
            print(f"\n📋 **{category}**")
            print("-" * 50)
            
            category_scores = []
            
            for query in queries:
                print(f"\n❓ Query: {query}")
                
                # Get search results
                results = self.search_entities(query, top_k=3)
                
                if results['ids'] and results['ids'][0]:
                    best_score = 1 - results['distances'][0][0]
                    best_entity = results['metadatas'][0][0]
                    
                    print(f"✅ Best Match: {best_entity.get('entity_name', best_entity.get('relationship_name', best_entity.get('subject_area', 'Unknown')))}")
                    print(f"📊 Score: {best_score:.3f}")
                    print(f"📂 Type: {best_entity['type'].title()}")
                    
                    category_scores.append(best_score)
                    all_scores.append(best_score)
                    
                    # Show quality assessment
                    if best_score > 0.3:
                        print("🎯 Quality: Excellent match")
                    elif best_score > 0.1:
                        print("👍 Quality: Good match")
                    elif best_score > -0.2:
                        print("⚠️  Quality: Fair match")
                    else:
                        print("❌ Quality: Poor match")
                else:
                    print("❌ No results found")
                    category_scores.append(-1.0)
                    all_scores.append(-1.0)
            
            # Category summary
            if category_scores:
                avg_score = sum(s for s in category_scores if s > -1) / len([s for s in category_scores if s > -1])
                print(f"\n📊 {category} Average Score: {avg_score:.3f}")
        
        # Overall summary
        print(f"\n" + "="*70)
        print("📊 OVERALL TEST RESULTS")
        print("="*70)
        
        valid_scores = [s for s in all_scores if s > -1]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            max_score = max(valid_scores)
            min_score = min(valid_scores)
            
            excellent_count = len([s for s in valid_scores if s > 0.3])
            good_count = len([s for s in valid_scores if 0.1 < s <= 0.3])
            fair_count = len([s for s in valid_scores if -0.2 < s <= 0.1])
            poor_count = len([s for s in valid_scores if s <= -0.2])
            
            print(f"📈 Average Score: {avg_score:.3f}")
            print(f"🏆 Best Score: {max_score:.3f}")
            print(f"📉 Lowest Score: {min_score:.3f}")
            print(f"📊 Total Queries: {len(valid_scores)}")
            print(f"\n🎯 Quality Distribution:")
            print(f"  • Excellent (>0.3): {excellent_count} queries")
            print(f"  • Good (0.1-0.3): {good_count} queries")
            print(f"  • Fair (-0.2-0.1): {fair_count} queries")
            print(f"  • Poor (<-0.2): {poor_count} queries")
            
            # Overall assessment
            if avg_score > 0.2:
                print(f"\n🎉 EXCELLENT: Your enterprise data model is performing very well!")
            elif avg_score > 0.0:
                print(f"\n👍 GOOD: Your enterprise data model is working well!")
            elif avg_score > -0.3:
                print(f"\n⚠️  FAIR: Your enterprise data model has reasonable performance.")
            else:
                print(f"\n❌ NEEDS IMPROVEMENT: Consider enhancing your data model descriptions.")
        
        print(f"\n✅ Test completed!")
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*70)
        print("🤖 Enhanced erwin Data Model Assistant")
        print("="*70)
        print("Ask me about your enterprise data model!")
        print("\nExamples:")
        print("• What is the Customer entity?")
        print("• What attributes does Order have?")
        print("• Tell me about Customer Management")
        print("• How are Product and OrderItem related?")
        print("\nCommands:")
        print("• 'test' - Run comprehensive test suite")
        print("• 'quit' - Exit the assistant")
        print("• 'help' - Show more examples")
        print("="*70)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye! Thanks for using the Enhanced erwin Assistant!")
                    break
                elif question.lower() == 'test':
                    self.run_test_queries()
                    continue
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
        examples = {
            "Entity Information": [
                "What is the Customer entity?",
                "Tell me about the Product entity",
                "Describe the Payment entity"
            ],
            "Attributes & Fields": [
                "What attributes does Customer have?",
                "Show me Product fields",
                "What data is in CustomerAddress?"
            ],
            "Business Rules": [
                "What are the business rules for Order?",
                "What validation exists for Customer?",
                "Tell me about Product constraints"
            ],
            "Subject Areas": [
                "Tell me about Customer Management",
                "What's in Product Catalog?",
                "Describe Financial Management"
            ],
            "Relationships": [
                "How are Customer and Order related?",
                "What connects Product to OrderItem?",
                "Show Order relationships"
            ]
        }
        
        print("\n💡 Example questions by category:")
        for category, examples_list in examples.items():
            print(f"\n📋 {category}:")
            for example in examples_list:
                print(f"  • {example}")

if __name__ == "__main__":
    # Create and start the enhanced chatbot
    chatbot = EnhancedErwinChatbot()
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Run comprehensive test suite")
    print("2. Start interactive chat")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        chatbot.run_test_queries()
        
        # Ask if they want to continue to chat
        continue_chat = input("\nWould you like to try interactive chat now? (y/n): ").strip().lower()
        if continue_chat in ['y', 'yes']:
            chatbot.chat_loop()
    else:
        chatbot.chat_loop()