"""
Load realistic enterprise erwin data model
"""
import chromadb
from sentence_transformers import SentenceTransformer

def get_realistic_erwin_data():
    """Complete enterprise e-commerce data model"""
    return {
        "model_info": {
            "name": "Enterprise E-Commerce Data Model",
            "version": "2.1",
            "description": "Complete data model for multi-channel e-commerce platform",
            "created_date": "2024-01-15",
            "last_modified": "2024-09-15"
        },
        "entities": [
            # Customer Management Subject Area
            {
                "name": "Customer",
                "business_name": "Customer",
                "description": """
                Central entity storing comprehensive customer information for the e-commerce platform.
                Supports both B2C and B2B customers with detailed profile management, preferences,
                and relationship tracking. Integrates with CRM, marketing automation, and support systems.
                """,
                "subject_area": "Customer Management",
                "attributes": [
                    "customer_id: Primary key, auto-generated unique identifier (INTEGER)",
                    "customer_type: B2C or B2B customer classification (VARCHAR(10))",
                    "email: Unique email address for login and communication (VARCHAR(255))",
                    "password_hash: Encrypted password for account security (VARCHAR(255))",
                    "first_name: Customer's given name (VARCHAR(100))",
                    "last_name: Customer's family name (VARCHAR(100))",
                    "company_name: Business name for B2B customers (VARCHAR(255))",
                    "phone_primary: Primary contact phone number (VARCHAR(20))",
                    "phone_secondary: Alternative contact number (VARCHAR(20))",
                    "date_of_birth: Customer birth date for age verification (DATE)",
                    "gender: Customer gender for demographics (VARCHAR(10))",
                    "date_registered: Account creation timestamp (TIMESTAMP)",
                    "date_last_login: Most recent login activity (TIMESTAMP)",
                    "email_verified: Email verification status (BOOLEAN)",
                    "phone_verified: Phone verification status (BOOLEAN)",
                    "marketing_opt_in: Permission for marketing communications (BOOLEAN)",
                    "loyalty_tier: Bronze, Silver, Gold, Platinum based on spend (VARCHAR(20))",
                    "loyalty_points: Current loyalty program points balance (INTEGER)",
                    "credit_limit: Maximum credit allowed for B2B customers (DECIMAL(10,2))",
                    "tax_exempt: Tax exemption status for organizations (BOOLEAN)",
                    "preferred_language: Customer's preferred language (VARCHAR(10))",
                    "timezone: Customer's timezone for scheduling (VARCHAR(50))",
                    "is_active: Account status indicator (BOOLEAN)",
                    "created_by: User who created the record (VARCHAR(100))",
                    "created_date: Record creation timestamp (TIMESTAMP)",
                    "modified_by: User who last modified the record (VARCHAR(100))",
                    "modified_date: Last modification timestamp (TIMESTAMP)"
                ],
                "business_rules": [
                    "Email addresses must be unique across all customer types",
                    "Phone numbers must follow E.164 international format",
                    "Passwords must meet complexity requirements (8+ chars, mixed case, numbers, symbols)",
                    "Date of birth required for age-restricted product purchases",
                    "Loyalty tier automatically calculated based on 12-month rolling purchase total",
                    "B2B customers require company name and tax ID",
                    "Marketing opt-in must be explicitly granted, not defaulted",
                    "Inactive customers cannot place orders or access premium features",
                    "Customer data retention follows GDPR requirements (7 years for tax, 3 years for marketing)",
                    "Credit limit requires manager approval for amounts over $10,000"
                ]
            },
            {
                "name": "CustomerAddress",
                "business_name": "Customer Address",
                "description": """
                Stores multiple addresses per customer including billing, shipping, and business addresses.
                Supports international address formats and address validation services.
                Tracks address usage history and preferences.
                """,
                "subject_area": "Customer Management",
                "attributes": [
                    "address_id: Primary key, unique address identifier (INTEGER)",
                    "customer_id: Foreign key to Customer entity (INTEGER)",
                    "address_type: Billing, Shipping, Business, Other (VARCHAR(20))",
                    "address_label: Customer-defined name for address (VARCHAR(100))",
                    "recipient_name: Name of person receiving at this address (VARCHAR(200))",
                    "company_name: Company name for business addresses (VARCHAR(255))",
                    "address_line1: Primary address line (VARCHAR(255))",
                    "address_line2: Secondary address line, apartment, suite (VARCHAR(255))",
                    "city: City or locality (VARCHAR(100))",
                    "state_province: State, province, or region (VARCHAR(100))",
                    "postal_code: ZIP code or postal code (VARCHAR(20))",
                    "country_code: ISO 3166-1 alpha-2 country code (VARCHAR(2))",
                    "latitude: GPS coordinate for delivery optimization (DECIMAL(10,8))",
                    "longitude: GPS coordinate for delivery optimization (DECIMAL(11,8))",
                    "is_default_billing: Default billing address flag (BOOLEAN)",
                    "is_default_shipping: Default shipping address flag (BOOLEAN)",
                    "is_validated: Address validation status (BOOLEAN)",
                    "validation_service: Service used for validation (VARCHAR(50))",
                    "delivery_instructions: Special delivery notes (TEXT)",
                    "is_active: Address availability status (BOOLEAN)"
                ],
                "business_rules": [
                    "Each customer must have at least one billing address",
                    "Only one default billing and shipping address per customer",
                    "Address validation required for shipping addresses",
                    "International addresses must include country code",
                    "Postal codes must match country format requirements",
                    "Business addresses require company name",
                    "Delivery instructions limited to 500 characters"
                ]
            },

            # Product Management Subject Area
            {
                "name": "Product",
                "business_name": "Product",
                "description": """
                Core product catalog entity containing detailed product information, pricing,
                inventory management, and merchandising data. Supports complex product hierarchies,
                variants, bundles, and configurable products.
                """,
                "subject_area": "Product Catalog",
                "attributes": [
                    "product_id: Primary key, unique product identifier (INTEGER)",
                    "parent_product_id: Foreign key for product variants/children (INTEGER)",
                    "sku: Stock keeping unit, unique across all products (VARCHAR(50))",
                    "upc: Universal product code for retail scanning (VARCHAR(20))",
                    "product_name: Display name for customers (VARCHAR(255))",
                    "product_description: Detailed product description (TEXT)",
                    "short_description: Brief product summary for listings (VARCHAR(500))",
                    "product_type: Simple, Configurable, Bundle, Virtual, Downloadable (VARCHAR(20))",
                    "category_id: Primary product category (INTEGER)",
                    "brand_id: Product brand reference (INTEGER)",
                    "vendor_id: Primary supplier reference (INTEGER)",
                    "manufacturer_part_number: Vendor's part number (VARCHAR(100))",
                    "weight: Product weight for shipping (DECIMAL(8,3))",
                    "weight_unit: Weight measurement unit (VARCHAR(10))",
                    "dimensions_length: Product length (DECIMAL(8,2))",
                    "dimensions_width: Product width (DECIMAL(8,2))",
                    "dimensions_height: Product height (DECIMAL(8,2))",
                    "dimension_unit: Dimension measurement unit (VARCHAR(10))",
                    "base_price: List price before discounts (DECIMAL(10,2))",
                    "cost_price: Product cost for margin calculation (DECIMAL(10,2))",
                    "msrp: Manufacturer suggested retail price (DECIMAL(10,2))",
                    "tax_class: Tax category for calculations (VARCHAR(50))",
                    "inventory_managed: Whether inventory is tracked (BOOLEAN)",
                    "stock_quantity: Current inventory level (INTEGER)",
                    "min_stock_level: Reorder point threshold (INTEGER)",
                    "max_stock_level: Maximum inventory to maintain (INTEGER)",
                    "backorders_allowed: Allow selling when out of stock (BOOLEAN)",
                    "track_inventory: Enable inventory tracking (BOOLEAN)",
                    "requires_shipping: Physical product requiring shipment (BOOLEAN)",
                    "is_digital: Digital/downloadable product (BOOLEAN)",
                    "download_limit: Max downloads for digital products (INTEGER)",
                    "download_expiry: Download link expiration days (INTEGER)",
                    "age_restriction: Minimum age required to purchase (INTEGER)",
                    "search_keywords: SEO and search terms (TEXT)",
                    "meta_title: Page title for SEO (VARCHAR(255))",
                    "meta_description: Meta description for SEO (VARCHAR(500))",
                    "is_featured: Featured product flag (BOOLEAN)",
                    "is_active: Product availability status (BOOLEAN)",
                    "date_available: Product availability start date (DATE)",
                    "date_discontinued: Product end-of-life date (DATE)"
                ],
                "business_rules": [
                    "SKU must be unique across all products and variants",
                    "Base price must be greater than cost price for profitable items",
                    "Weight required for all shippable products",
                    "Dimensions required for oversized shipping calculations",
                    "Digital products cannot have physical inventory",
                    "Age-restricted products require customer age verification",
                    "Featured products must be active and available",
                    "Discontinued products cannot be reactivated",
                    "Min stock level must be less than max stock level",
                    "Backorders only allowed for non-digital products",
                    "Download limits apply only to digital products"
                ]
            },
            {
                "name": "ProductCategory",
                "business_name": "Product Category",
                "description": """
                Hierarchical product categorization system supporting multiple levels of organization.
                Enables navigation, filtering, merchandising, and reporting by product groupings.
                """,
                "subject_area": "Product Catalog",
                "attributes": [
                    "category_id: Primary key, unique category identifier (INTEGER)",
                    "parent_category_id: Foreign key for category hierarchy (INTEGER)",
                    "category_name: Display name for the category (VARCHAR(255))",
                    "category_description: Detailed category description (TEXT)",
                    "category_path: Full hierarchical path (VARCHAR(1000))",
                    "level: Hierarchy level (0=root, 1=main, 2=sub, etc.) (INTEGER)",
                    "sort_order: Display order within parent category (INTEGER)",
                    "image_url: Category image for navigation (VARCHAR(500))",
                    "seo_url: SEO-friendly URL slug (VARCHAR(255))",
                    "meta_title: Page title for SEO (VARCHAR(255))",
                    "meta_description: Meta description for SEO (VARCHAR(500))",
                    "is_active: Category visibility status (BOOLEAN)",
                    "show_in_menu: Display in navigation menu (BOOLEAN)",
                    "product_count: Number of products in category (INTEGER)"
                ],
                "business_rules": [
                    "Root categories have no parent (parent_category_id = NULL)",
                    "Category names must be unique within the same parent",
                    "SEO URLs must be unique across all categories",
                    "Categories cannot be their own parent (circular reference check)",
                    "Inactive categories hide all child categories and products",
                    "Category deletion requires moving or deleting all child categories",
                    "Product count automatically updated when products added/removed"
                ]
            },

            # Order Management Subject Area
            {
                "name": "Order",
                "business_name": "Sales Order",
                "description": """
                Comprehensive order management entity tracking the complete purchase lifecycle
                from cart to delivery. Supports multiple order types, payment methods, and
                fulfillment scenarios including split shipments and backorders.
                """,
                "subject_area": "Order Management",
                "attributes": [
                    "order_id: Primary key, unique order identifier (INTEGER)",
                    "order_number: Human-readable order reference (VARCHAR(50))",
                    "customer_id: Foreign key to Customer entity (INTEGER)",
                    "order_date: Order placement timestamp (TIMESTAMP)",
                    "order_type: Standard, Subscription, Return, Exchange (VARCHAR(20))",
                    "order_source: Web, Mobile, Phone, In-Store, API (VARCHAR(20))",
                    "order_status: Pending, Processing, Shipped, Delivered, Cancelled (VARCHAR(20))",
                    "payment_status: Pending, Authorized, Captured, Refunded (VARCHAR(20))",
                    "fulfillment_status: Pending, Partial, Complete, Cancelled (VARCHAR(20))",
                    "currency_code: ISO 4217 currency code (VARCHAR(3))",
                    "exchange_rate: Rate if different from base currency (DECIMAL(10,6))",
                    "subtotal: Order total before taxes and fees (DECIMAL(12,2))",
                    "tax_amount: Total tax charged (DECIMAL(12,2))",
                    "shipping_amount: Shipping and handling charges (DECIMAL(12,2))",
                    "discount_amount: Total discounts applied (DECIMAL(12,2))",
                    "total_amount: Final order total (DECIMAL(12,2))",
                    "billing_address_id: Foreign key to CustomerAddress (INTEGER)",
                    "shipping_address_id: Foreign key to CustomerAddress (INTEGER)",
                    "shipping_method: Delivery method selected (VARCHAR(100))",
                    "tracking_number: Shipment tracking reference (VARCHAR(100))",
                    "expected_delivery_date: Estimated delivery date (DATE)",
                    "actual_delivery_date: Confirmed delivery date (DATE)",
                    "gift_message: Customer gift message (TEXT)",
                    "internal_notes: Staff notes and comments (TEXT)",
                    "coupon_code: Applied discount code (VARCHAR(50))",
                    "loyalty_points_used: Points redeemed for discount (INTEGER)",
                    "loyalty_points_earned: Points earned from purchase (INTEGER)",
                    "referral_source: How customer found the store (VARCHAR(100))",
                    "sales_rep_id: Assigned sales representative (INTEGER)",
                    "priority_level: Order processing priority 1-5 (INTEGER)",
                    "requires_signature: Delivery signature required (BOOLEAN)",
                    "is_gift: Order marked as gift (BOOLEAN)",
                    "cancel_reason: Reason for cancellation (VARCHAR(255))",
                    "cancelled_by: User who cancelled the order (VARCHAR(100))",
                    "cancelled_date: Cancellation timestamp (TIMESTAMP)"
                ],
                "business_rules": [
                    "Order number must be unique and sequential",
                    "Total amount equals subtotal + tax + shipping - discounts",
                    "Order status workflow: Pending → Processing → Shipped → Delivered",
                    "Payment must be authorized before order processing",
                    "Shipped orders cannot be cancelled, only returned",
                    "Gift orders require separate billing and shipping addresses",
                    "Loyalty points usage cannot exceed customer balance",
                    "International orders require additional documentation",
                    "Cancelled orders release inventory back to available stock",
                    "Delivery signature required for orders over $500"
                ]
            },
            {
                "name": "OrderItem",
                "business_name": "Order Line Item",
                "description": """
                Individual product line items within an order. Tracks quantity, pricing,
                discounts, and fulfillment status for each product purchased. Supports
                partial fulfillment and backorder management.
                """,
                "subject_area": "Order Management",
                "attributes": [
                    "order_item_id: Primary key, unique line item identifier (INTEGER)",
                    "order_id: Foreign key to Order entity (INTEGER)",
                    "product_id: Foreign key to Product entity (INTEGER)",
                    "product_sku: Product SKU at time of order (VARCHAR(50))",
                    "product_name: Product name at time of order (VARCHAR(255))",
                    "quantity_ordered: Number of items ordered (INTEGER)",
                    "quantity_shipped: Number of items shipped (INTEGER)",
                    "quantity_cancelled: Number of items cancelled (INTEGER)",
                    "unit_price: Price per item at time of order (DECIMAL(10,2))",
                    "discount_percent: Percentage discount applied (DECIMAL(5,2))",
                    "discount_amount: Dollar amount discount applied (DECIMAL(10,2))",
                    "tax_percent: Tax rate applied (DECIMAL(5,3))",
                    "tax_amount: Tax amount for this line item (DECIMAL(10,2))",
                    "line_total: Total for this line item (DECIMAL(12,2))",
                    "cost_price: Product cost at time of order (DECIMAL(10,2))",
                    "margin_amount: Profit margin (line_total - cost) (DECIMAL(10,2))",
                    "backorder_quantity: Quantity on backorder (INTEGER)",
                    "expected_ship_date: Expected shipment date (DATE)",
                    "actual_ship_date: Actual shipment date (DATE)",
                    "serial_numbers: Product serial numbers if applicable (TEXT)",
                    "personalization_text: Custom text for personalized items (TEXT)",
                    "gift_wrap: Gift wrapping option selected (VARCHAR(50))",
                    "special_instructions: Customer instructions (TEXT)"
                ],
                "business_rules": [
                    "Quantity ordered must be greater than zero",
                    "Quantity shipped cannot exceed quantity ordered",
                    "Line total equals (unit_price * quantity) - discount + tax",
                    "Cancelled quantity reduces from ordered quantity",
                    "Backorder quantity cannot exceed ordered - shipped - cancelled",
                    "Serial numbers required for serialized products",
                    "Personalization adds to processing time",
                    "Gift wrap incurs additional charges"
                ]
            },

            # Inventory Management Subject Area
            {
                "name": "Inventory",
                "business_name": "Inventory",
                "description": """
                Real-time inventory tracking across multiple warehouses and locations.
                Manages stock levels, reservations, allocations, and movement history
                with full audit trail and automated reorder capabilities.
                """,
                "subject_area": "Inventory Management",
                "attributes": [
                    "inventory_id: Primary key, unique inventory record (INTEGER)",
                    "product_id: Foreign key to Product entity (INTEGER)",
                    "warehouse_id: Foreign key to Warehouse entity (INTEGER)",
                    "location_code: Specific warehouse location (VARCHAR(20))",
                    "quantity_on_hand: Physical inventory count (INTEGER)",
                    "quantity_available: Available for sale (on_hand - reserved) (INTEGER)",
                    "quantity_reserved: Reserved for pending orders (INTEGER)",
                    "quantity_on_order: Incoming from suppliers (INTEGER)",
                    "quantity_allocated: Allocated to orders (INTEGER)",
                    "reorder_point: Automatic reorder threshold (INTEGER)",
                    "reorder_quantity: Standard reorder amount (INTEGER)",
                    "max_stock_level: Maximum inventory to maintain (INTEGER)",
                    "last_count_date: Most recent physical count (DATE)",
                    "last_received_date: Most recent receipt (DATE)",
                    "average_cost: Weighted average cost (DECIMAL(10,2))",
                    "last_cost: Most recent purchase cost (DECIMAL(10,2))",
                    "valuation_method: FIFO, LIFO, Average Cost (VARCHAR(20))",
                    "abc_classification: A, B, C classification for importance (VARCHAR(1))",
                    "cycle_count_frequency: Days between counts (INTEGER)",
                    "is_serialized: Product requires serial tracking (BOOLEAN)",
                    "is_lot_tracked: Product requires lot/batch tracking (BOOLEAN)",
                    "expiration_date: Product expiration if applicable (DATE)",
                    "quarantine_quantity: Items on quality hold (INTEGER)",
                    "damaged_quantity: Damaged inventory count (INTEGER)"
                ],
                "business_rules": [
                    "Quantity available cannot be negative",
                    "Reserved quantity cannot exceed on-hand quantity",
                    "Reorder point must be less than max stock level",
                    "Serialized products require individual tracking",
                    "Lot-tracked products require batch information",
                    "Expired products cannot be sold",
                    "Quarantined inventory not available for sale",
                    "Cycle count frequency based on ABC classification"
                ]
            },

            # Financial Management Subject Area
            {
                "name": "Payment",
                "business_name": "Payment Transaction",
                "description": """
                Payment processing and transaction management supporting multiple payment
                methods, currencies, and processors. Tracks authorization, capture,
                refund, and chargeback processes with full PCI compliance.
                """,
                "subject_area": "Financial Management",
                "attributes": [
                    "payment_id: Primary key, unique payment identifier (INTEGER)",
                    "order_id: Foreign key to Order entity (INTEGER)",
                    "payment_method: Credit Card, PayPal, Bank Transfer, etc. (VARCHAR(50))",
                    "payment_processor: Stripe, PayPal, Square, etc. (VARCHAR(50))",
                    "transaction_id: Processor transaction reference (VARCHAR(100))",
                    "authorization_code: Payment authorization code (VARCHAR(50))",
                    "payment_status: Pending, Authorized, Captured, Failed, Refunded (VARCHAR(20))",
                    "currency_code: ISO 4217 currency code (VARCHAR(3))",
                    "amount: Payment amount in specified currency (DECIMAL(12,2))",
                    "fee_amount: Processing fees charged (DECIMAL(10,2))",
                    "net_amount: Amount after fees (DECIMAL(12,2))",
                    "exchange_rate: Currency conversion rate (DECIMAL(10,6))",
                    "card_type: Visa, MasterCard, Amex, etc. (VARCHAR(20))",
                    "card_last_four: Last 4 digits of card (VARCHAR(4))",
                    "card_expiry_month: Card expiration month (INTEGER)",
                    "card_expiry_year: Card expiration year (INTEGER)",
                    "billing_address_match: Address verification result (BOOLEAN)",
                    "cvv_match: CVV verification result (BOOLEAN)",
                    "risk_score: Fraud risk assessment 0-100 (INTEGER)",
                    "gateway_response: Detailed processor response (TEXT)",
                    "authorization_date: Initial authorization timestamp (TIMESTAMP)",
                    "capture_date: Payment capture timestamp (TIMESTAMP)",
                    "settlement_date: Funds settlement date (DATE)",
                    "refund_amount: Total amount refunded (DECIMAL(12,2))",
                    "refund_reason: Reason for refund (VARCHAR(255))",
                    "chargeback_amount: Chargeback amount if applicable (DECIMAL(12,2))",
                    "chargeback_reason: Chargeback reason code (VARCHAR(100))",
                    "is_recurring: Recurring payment flag (BOOLEAN)",
                    "recurring_frequency: Payment frequency (VARCHAR(20))"
                ],
                "business_rules": [
                    "Payment amount must match order total",
                    "Authorization required before capture",
                    "Refunds cannot exceed captured amount",
                    "High-risk transactions require manual review",
                    "PCI compliance required for card data handling",
                    "Failed payments trigger retry logic",
                    "Chargebacks initiate dispute process",
                    "Recurring payments require customer consent"
                ]
            }
        ],
        "relationships": [
            {
                "name": "Customer_Has_Addresses",
                "parent_entity": "Customer",
                "child_entity": "CustomerAddress",
                "relationship_type": "One-to-Many",
                "description": "Each customer can have multiple addresses for billing and shipping"
            },
            {
                "name": "Customer_Places_Orders",
                "parent_entity": "Customer", 
                "child_entity": "Order",
                "relationship_type": "One-to-Many",
                "description": "Customers can place multiple orders over time"
            },
            {
                "name": "Order_Contains_Items",
                "parent_entity": "Order",
                "child_entity": "OrderItem", 
                "relationship_type": "One-to-Many",
                "description": "Each order contains one or more line items"
            },
            {
                "name": "Product_Ordered_As_Items",
                "parent_entity": "Product",
                "child_entity": "OrderItem",
                "relationship_type": "One-to-Many", 
                "description": "Products can appear in multiple order line items"
            },
            {
                "name": "Category_Contains_Products",
                "parent_entity": "ProductCategory",
                "child_entity": "Product",
                "relationship_type": "One-to-Many",
                "description": "Product categories contain multiple products"
            },
            {
                "name": "Product_Has_Inventory",
                "parent_entity": "Product",
                "child_entity": "Inventory",
                "relationship_type": "One-to-Many",
                "description": "Products have inventory records across multiple locations"
            },
            {
                "name": "Order_Has_Payments",
                "parent_entity": "Order",
                "child_entity": "Payment",
                "relationship_type": "One-to-Many",
                "description": "Orders can have multiple payment transactions"
            }
        ]
    }

class RealisticErwinLoader:
    def __init__(self):
        print("🏗️ Initializing Realistic erwin Data Loader...")
        
        # Set up ChromaDB
        self.client = chromadb.PersistentClient(path="./enterprise_chroma_db")
        
        # Create new collection for enterprise data
        try:
            self.client.delete_collection("enterprise-erwin-model")
        except:
            pass
            
        self.collection = self.client.create_collection("enterprise-erwin-model")
        
        # Set up embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("✅ Realistic data loader ready!")
    
    def load_data(self):
        """Load the complete enterprise data model"""
        print("\n📥 Loading realistic enterprise erwin data...")
        
        data = get_realistic_erwin_data()
        
        # Load model metadata
        model_info = data['model_info']
        print(f"📋 Loading model: {model_info['name']} v{model_info['version']}")
        
        # Load entities
        entities_by_subject = {}
        for entity in data['entities']:
            subject_area = entity['subject_area']
            if subject_area not in entities_by_subject:
                entities_by_subject[subject_area] = []
            entities_by_subject[subject_area].append(entity)
        
        print(f"\n📊 Model Statistics:")
        print(f"  • Total Entities: {len(data['entities'])}")
        print(f"  • Subject Areas: {len(entities_by_subject)}")
        print(f"  • Relationships: {len(data['relationships'])}")
        
        for subject_area, entities in entities_by_subject.items():
            print(f"  • {subject_area}: {len(entities)} entities")
        
        # Add entities to ChromaDB
        print(f"\n📥 Adding entities to ChromaDB...")
        for i, entity in enumerate(data['entities'], 1):
            self._add_entity(entity, i, len(data['entities']))
        
        # Add relationships
        print(f"\n🔗 Adding relationships...")
        for i, relationship in enumerate(data['relationships'], 1):
            self._add_relationship(relationship, i, len(data['relationships']))
        
        # Add subject area summaries
        print(f"\n📂 Adding subject area summaries...")
        for subject_area, entities in entities_by_subject.items():
            self._add_subject_area_summary(subject_area, entities)
        
        print(f"\n✅ Successfully loaded complete enterprise data model!")
        
        # Show final statistics
        stats = self.get_collection_stats()
        print(f"\n📊 Final Collection Statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    
    def _add_entity(self, entity, current, total):
        """Add individual entity to ChromaDB"""
        print(f"  [{current:2d}/{total}] Adding {entity['name']} ({entity['subject_area']})")
        
        # Create comprehensive text representation
        text_content = f"""
        Entity: {entity['name']}
        Business Name: {entity['business_name']}
        Subject Area: {entity['subject_area']}
        
        Description: {entity['description']}
        
        Attributes ({len(entity['attributes'])} total):
        {chr(10).join(['• ' + attr for attr in entity['attributes']])}
        
        Business Rules ({len(entity['business_rules'])} total):
        {chr(10).join(['• ' + rule for rule in entity['business_rules']])}
        """
        
        # Create embedding
        embedding = self.embedding_model.encode(text_content)
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_content],
            metadatas={
                "type": "entity",
                "entity_name": entity['name'],
                "business_name": entity['business_name'],
                "subject_area": entity['subject_area'],
                "attribute_count": len(entity['attributes']),
                "rule_count": len(entity['business_rules'])
            },
            ids=[f"entity_{entity['name'].lower().replace(' ', '_')}"]
        )
    
    def _add_relationship(self, relationship, current, total):
        """Add relationship to ChromaDB"""
        print(f"  [{current:2d}/{total}] Adding relationship: {relationship['name']}")
        
        text_content = f"""
        Relationship: {relationship['name']}
        Type: {relationship['relationship_type']}
        Parent Entity: {relationship['parent_entity']}
        Child Entity: {relationship['child_entity']}
        
        Description: {relationship['description']}
        
        This relationship connects {relationship['parent_entity']} to {relationship['child_entity']} 
        in a {relationship['relationship_type']} relationship, meaning {relationship['description']}
        """
        
        embedding = self.embedding_model.encode(text_content)
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_content],
            metadatas={
                "type": "relationship",
                "relationship_name": relationship['name'],
                "parent_entity": relationship['parent_entity'],
                "child_entity": relationship['child_entity'],
                "relationship_type": relationship['relationship_type']
            },
            ids=[f"relationship_{relationship['name'].lower().replace(' ', '_')}"]
        )
    
    def _add_subject_area_summary(self, subject_area, entities):
        """Add subject area overview to ChromaDB"""
        print(f"  📂 Adding summary for {subject_area}")
        
        entity_names = [entity['name'] for entity in entities]
        total_attributes = sum(len(entity['attributes']) for entity in entities)
        total_rules = sum(len(entity['business_rules']) for entity in entities)
        
        text_content = f"""
        Subject Area: {subject_area}
        
        Overview: The {subject_area} subject area contains {len(entities)} entities that work together 
        to manage and track information related to {subject_area.lower()} operations.
        
        Entities in this Subject Area:
        {chr(10).join(['• ' + name for name in entity_names])}
        
        Total Attributes: {total_attributes}
        Total Business Rules: {total_rules}
        
        This subject area handles all data requirements for {subject_area.lower()} including 
        data storage, business logic, and operational requirements.
        """
        
        embedding = self.embedding_model.encode(text_content)
        
        self.collection.add(
            embeddings=[embedding.tolist()],
            documents=[text_content],
            metadatas={
                "type": "subject_area",
                "subject_area": subject_area,
                "entity_count": len(entities),
                "total_attributes": total_attributes,
                "total_rules": total_rules
            },
            ids=[f"subject_area_{subject_area.lower().replace(' ', '_')}"]
        )
    
    def get_collection_stats(self):
        """Get detailed collection statistics"""
        try:
            all_data = self.collection.get(include=["metadatas"])
            
            stats = {
                "Total Documents": len(all_data['metadatas']),
                "Entities": len([m for m in all_data['metadatas'] if m['type'] == 'entity']),
                "Relationships": len([m for m in all_data['metadatas'] if m['type'] == 'relationship']),
                "Subject Areas": len([m for m in all_data['metadatas'] if m['type'] == 'subject_area'])
            }
            
            # Count by subject area
            subject_areas = {}
            for metadata in all_data['metadatas']:
                if metadata['type'] == 'entity':
                    area = metadata['subject_area']
                    if area not in subject_areas:
                        subject_areas[area] = 0
                    subject_areas[area] += 1
            
            stats.update(subject_areas)
            return stats
            
        except Exception as e:
            return {"Error": str(e)}

if __name__ == "__main__":
    print("🚀 Starting Realistic erwin Data Load...")
    print("=" * 60)
    
    loader = RealisticErwinLoader()
    loader.load_data()
    
    print("\n" + "=" * 60)
    print("🎉 Realistic data loading completed!")
    print("\nNext steps:")
    print("1. Run: python enhanced_erwin_chat.py")
    print("2. Try asking about specific subject areas")
    print("3. Ask about entity relationships")
    print("4. Explore business rules and attributes")