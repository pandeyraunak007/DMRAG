"""
Improved ChromaDB test with better erwin data
"""
import chromadb
from sentence_transformers import SentenceTransformer

print("🚀 Starting improved ChromaDB test...")

# Step 1: Set up ChromaDB
print("📁 Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./improved_chroma_db")
collection = client.get_or_create_collection("erwin-entities")
print("✅ ChromaDB ready!")

# Step 2: Set up embeddings
print("🧠 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model ready!")

# Step 3: Better erwin sample data (more detailed)
sample_entities = [
    {
        "name": "Customer",
        "description": """
        The Customer entity stores comprehensive information about registered users of the e-commerce platform.
        This includes personal details, contact information, account preferences, and loyalty program data.
        Customers can place multiple orders and have a complete profile for personalized experiences.
        """,
        "attributes": [
            "customer_id: Primary key, unique identifier for each customer",
            "email: Customer's email address, used for login and communication",
            "first_name: Customer's first name for personalization",
            "last_name: Customer's family name for formal communication",
            "phone: Contact telephone number for support and notifications",
            "date_registered: When the customer account was created",
            "loyalty_tier: Bronze, Silver, Gold based on purchase history",
            "is_active: Whether the customer account is currently active"
        ],
        "business_rules": [
            "Email addresses must be unique across all customers",
            "Phone numbers must follow international format validation",
            "Loyalty tier is automatically calculated based on total purchase amount",
            "Inactive customers cannot place new orders",
            "Customer data must be retained for 7 years for compliance"
        ],
        "subject_area": "Customer Management"
    },
    {
        "name": "Order",
        "description": """
        The Order entity represents purchase transactions made by customers.
        It tracks the complete order lifecycle from placement to delivery,
        including order status, payment information, and shipping details.
        Each order links to a customer and contains multiple order items.
        """,
        "attributes": [
            "order_id: Primary key, unique identifier for each order",
            "customer_id: Foreign key linking to the Customer entity",
            "order_date: Timestamp when the order was placed",
            "order_status: Current status - Pending, Processing, Shipped, Delivered, Cancelled",
            "subtotal: Order amount before taxes and shipping",
            "tax_amount: Calculated tax based on customer location",
            "shipping_cost: Delivery charges for the order",
            "total_amount: Final amount including all charges",
            "shipping_address_id: Reference to delivery address"
        ],
        "business_rules": [
            "Order status must follow proper workflow: Pending → Processing → Shipped → Delivered",
            "Total amount must equal subtotal plus tax plus shipping",
            "Orders cannot be cancelled after shipping",
            "Customer must be active to place orders",
            "Order date cannot be in the future"
        ],
        "subject_area": "Order Management"
    },
    {
        "name": "Product",
        "description": """
        The Product entity contains information about items available for purchase.
        This includes product details, pricing, inventory levels, and categorization.
        Products can be part of multiple orders and have complex pricing rules.
        """,
        "attributes": [
            "product_id: Primary key, unique identifier for each product",
            "sku: Stock keeping unit, unique code for inventory tracking",
            "product_name: Display name shown to customers",
            "description: Detailed product description for marketing",
            "category_id: Foreign key to Product Category",
            "price: Current selling price to customers",
            "cost: Internal cost for margin calculation",
            "inventory_count: Current stock level available",
            "is_active: Whether product is currently available for sale"
        ],
        "business_rules": [
            "SKU must be unique across all products",
            "Selling price must be greater than cost price",
            "Inventory count cannot be negative",
            "Inactive products cannot be ordered",
            "Price changes require manager approval"
        ],
        "subject_area": "Product Catalog"
    }
]

# Step 4: Clear existing data and add improved data
print("\n🧹 Clearing existing data...")
try:
    existing_data = collection.get()
    if existing_data['ids']:
        collection.delete(ids=existing_data['ids'])
        print("✅ Cleared old data")
except:
    print("📝 No existing data to clear")

print("\n📥 Adding detailed entities...")
for entity in sample_entities:
    # Create comprehensive text description
    text = f"""
    Entity Name: {entity['name']}
    Subject Area: {entity['subject_area']}
    
    Description: {entity['description']}
    
    Attributes:
    {chr(10).join(entity['attributes'])}
    
    Business Rules:
    {chr(10).join(entity['business_rules'])}
    """
    
    # Create embedding
    embedding = embedding_model.encode(text)
    
    # Add to ChromaDB
    collection.add(
        embeddings=[embedding.tolist()],
        documents=[text],
        metadatas={
            "entity_name": entity['name'],
            "subject_area": entity['subject_area'],
            "attribute_count": len(entity['attributes']),
            "rule_count": len(entity['business_rules'])
        },
        ids=[f"entity_{entity['name'].lower()}"]
    )
    print(f"✅ Added {entity['name']} entity ({entity['subject_area']})")

# Step 5: Test with better queries
print("\n🔍 Testing improved search...")
test_queries = [
    "customer information and personal data",
    "order processing and transaction management", 
    "business rules and validation requirements",
    "product catalog and inventory management",
    "email and contact information",
    "pricing and cost calculations"
]

for query in test_queries:
    print(f"\n--- Searching for: '{query}' ---")
    
    # Create query embedding
    query_embedding = embedding_model.encode(query)
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3,
        include=["metadatas", "documents", "distances"]
    )
    
    # Show results
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            entity_name = results['metadatas'][0][i]['entity_name']
            subject_area = results['metadatas'][0][i]['subject_area']
            score = 1 - results['distances'][0][i]
            print(f"  📊 Found: {entity_name} ({subject_area}) - Score: {score:.3f}")
    else:
        print("  📝 No results found")

# Step 6: Show collection stats
print(f"\n📊 Collection Statistics:")
try:
    count = collection.count()
    print(f"  Total entities: {count}")
    
    # Get all entities
    all_data = collection.get(include=["metadatas"])
    subject_areas = {}
    for metadata in all_data['metadatas']:
        area = metadata['subject_area']
        if area not in subject_areas:
            subject_areas[area] = 0
        subject_areas[area] += 1
    
    print(f"  Subject areas: {dict(subject_areas)}")
    
except Exception as e:
    print(f"  Error getting stats: {e}")

print("\n🎉 Improved test completed successfully!")
print("✅ ChromaDB is working with detailed erwin data!")