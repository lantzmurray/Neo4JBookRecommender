#!/usr/bin/env python3
"""
Test Neo4j connection with cloud credentials
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
load_dotenv()

# Get credentials
uri = os.getenv('NEO4J_URI')
username = os.getenv('NEO4J_USERNAME')
password = os.getenv('NEO4J_PASSWORD')

print(f"Connecting to: {uri}")
print(f"Username: {username}")
print(f"Password: {'*' * len(password) if password else 'None'}")

try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    # Test connection
    with driver.session() as session:
        result = session.run("RETURN 'Hello, Neo4j!' as message")
        record = result.single()
        print(f"Connection successful! Message: {record['message']}")
        
        # Check if database is empty or has data
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        count = result.single()['node_count']
        print(f"Total nodes in database: {count}")
        
    driver.close()
    print("Connection test completed successfully!")
    
except Exception as e:
    print(f"Connection failed: {e}")
    print(f"Error type: {type(e).__name__}")