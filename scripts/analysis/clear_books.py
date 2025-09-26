#!/usr/bin/env python3
"""
Clear existing book data from Neo4j database
"""

from dotenv import load_dotenv
from neo4j import GraphDatabase
import os

# Load environment variables
load_dotenv()

# Connect to Neo4j
driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'), 
    auth=(os.getenv('NEO4J_USERNAME'), os.getenv('NEO4J_PASSWORD'))
)

try:
    with driver.session() as session:
        # Clear existing book data with bookrecommender tag
        result = session.run('MATCH (n) WHERE n.tag = "bookrecommender" DETACH DELETE n')
        print('Cleared existing book data with bookrecommender tag')
        
        # Get count of remaining nodes
        count_result = session.run('MATCH (n) RETURN count(n) as total')
        total_nodes = count_result.single()['total']
        print(f'Total nodes remaining in database: {total_nodes}')
        
finally:
    driver.close()