#!/usr/bin/env python3
"""
Script to change Neo4j default password using Python driver
"""

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def change_password():
    """Change Neo4j default password"""
    try:
        # Connect with default credentials
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4j'))
        
        with driver.session(database='system') as session:
            # Change password
            result = session.run("ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'bookrecommender'")
            logger.info("Password changed successfully!")
            
        driver.close()
        
        # Test new connection
        new_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'bookrecommender'))
        with new_driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            if test_value == 1:
                logger.info("New password verified successfully!")
        
        new_driver.close()
        
    except Exception as e:
        logger.error(f"Error changing password: {e}")

if __name__ == "__main__":
    change_password()