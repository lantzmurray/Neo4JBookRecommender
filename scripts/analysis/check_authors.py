#!/usr/bin/env python3
"""
Check existing Author nodes in Neo4j database
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
        # Check for Stephen King authors
        result = session.run('MATCH (a:Author {name: "Stephen King"}) RETURN a.name, a.tag')
        authors = [record for record in result]
        print('Stephen King authors:')
        for author in authors:
            print(f'- Name: {author["a.name"]}, Tag: {author["a.tag"]}')
        
        # Check total author count
        count_result = session.run('MATCH (a:Author) RETURN count(a) as total')
        total_authors = count_result.single()['total']
        print(f'Total authors in database: {total_authors}')
        
        # Check authors without bookrecommender tag
        no_tag_result = session.run('MATCH (a:Author) WHERE a.tag IS NULL OR a.tag <> "bookrecommender" RETURN count(a) as total')
        no_tag_authors = no_tag_result.single()['total']
        print(f'Authors without bookrecommender tag: {no_tag_authors}')
        
finally:
    driver.close()