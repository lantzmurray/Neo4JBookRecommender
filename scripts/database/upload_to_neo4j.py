#!/usr/bin/env python3
"""
Neo4j Upload Script for Book Recommendation System
Uploads books with embeddings to Neo4j with 'bookrecommender' tags
Verifies existing Xerox documents are preserved
"""

import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logging.warning("Neo4j driver not available. Install with: pip install neo4j")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j configuration - Use local instance
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j')

class Neo4jBookUploader:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j driver not available. Install with: pip install neo4j")
        
        # Use local Neo4j instance with updated password
        self.driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'bookrecommender'))
        self.verify_connection()
    
    def verify_connection(self):
        """Verify Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Neo4j connection successful")
                else:
                    raise Exception("Connection test failed")
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise
    
    def check_xerox_documents(self) -> Dict[str, Any]:
        """Check for existing Xerox documents in the database"""
        logger.info("Checking for existing Xerox documents...")
        
        with self.driver.session() as session:
            # Check for nodes with xerox-related tags or properties
            xerox_queries = [
                "MATCH (n) WHERE any(label IN labels(n) WHERE label CONTAINS 'xerox' OR label CONTAINS 'Xerox') RETURN count(n) as xerox_labeled_nodes",
                "MATCH (n) WHERE any(key IN keys(n) WHERE key CONTAINS 'xerox' OR key CONTAINS 'Xerox') RETURN count(n) as xerox_property_nodes",
                "MATCH (n) WHERE any(value IN [prop IN keys(n) | n[prop]] WHERE toString(value) CONTAINS 'xerox' OR toString(value) CONTAINS 'Xerox') RETURN count(n) as xerox_content_nodes",
                "MATCH (n:Document) RETURN count(n) as document_nodes",
                "MATCH (n) WHERE n.source CONTAINS 'xerox' OR n.source CONTAINS 'Xerox' RETURN count(n) as xerox_source_nodes"
            ]
            
            xerox_info = {}
            
            for query in xerox_queries:
                try:
                    result = session.run(query)
                    record = result.single()
                    if record:
                        key = list(record.keys())[0]
                        xerox_info[key] = record[key]
                except Exception as e:
                    logger.warning(f"Query failed: {query[:50]}... Error: {e}")
            
            # Get sample of existing nodes to understand structure
            try:
                sample_query = "MATCH (n) RETURN labels(n) as labels, keys(n) as properties LIMIT 10"
                result = session.run(sample_query)
                sample_nodes = [{"labels": record["labels"], "properties": record["properties"]} 
                              for record in result]
                xerox_info["sample_nodes"] = sample_nodes
            except Exception as e:
                logger.warning(f"Could not get sample nodes: {e}")
                xerox_info["sample_nodes"] = []
            
            # Get total node count
            try:
                total_result = session.run("MATCH (n) RETURN count(n) as total_nodes")
                xerox_info["total_nodes"] = total_result.single()["total_nodes"]
            except Exception as e:
                logger.warning(f"Could not get total node count: {e}")
                xerox_info["total_nodes"] = 0
        
        return xerox_info
    
    def create_indexes_and_constraints(self):
        """Create necessary indexes and constraints for book recommendation system"""
        logger.info("Creating indexes and constraints...")
        
        with self.driver.session() as session:
            # Indexes and constraints for book recommendation system
            index_queries = [
                # Book indexes
                "CREATE INDEX book_id_index IF NOT EXISTS FOR (b:Book) ON (b.book_id)",
                "CREATE INDEX book_title_index IF NOT EXISTS FOR (b:Book) ON (b.title)",
                "CREATE INDEX book_isbn_index IF NOT EXISTS FOR (b:Book) ON (b.isbn)",
                "CREATE INDEX book_tag_index IF NOT EXISTS FOR (b:Book) ON (b.tag)",
                "CREATE INDEX book_source_index IF NOT EXISTS FOR (b:Book) ON (b.source)",
                
                # Author indexes
                "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
                "CREATE INDEX author_tag_index IF NOT EXISTS FOR (a:Author) ON (a.tag)",
                
                # Genre indexes
                "CREATE INDEX genre_name_index IF NOT EXISTS FOR (g:Genre) ON (g.name)",
                "CREATE INDEX genre_tag_index IF NOT EXISTS FOR (g:Genre) ON (g.tag)",
                
                # Publisher indexes
                "CREATE INDEX publisher_name_index IF NOT EXISTS FOR (p:Publisher) ON (p.name)",
                "CREATE INDEX publisher_tag_index IF NOT EXISTS FOR (p:Publisher) ON (p.tag)",
                
                # Vector index for similarity search
                "CREATE VECTOR INDEX book_embeddings_index IF NOT EXISTS FOR (b:Book) ON (b.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}"
            ]
            
            for query in index_queries:
                try:
                    session.run(query)
                    logger.info(f"✅ Executed: {query[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️  Index creation failed: {query[:50]}... Error: {e}")
    
    def upload_books_batch(self, books: List[Dict[str, Any]], batch_size: int = 100):
        """Upload books in batches to Neo4j"""
        logger.info(f"Uploading {len(books)} books in batches of {batch_size}...")
        
        total_batches = (len(books) - 1) // batch_size + 1
        
        for batch_idx in range(0, len(books), batch_size):
            batch = books[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} books)")
            
            with self.driver.session() as session:
                # Create books and relationships in a single transaction
                session.execute_write(self._create_books_transaction, batch)
    
    def _create_books_transaction(self, tx, books: List[Dict[str, Any]]):
        """Transaction to create books and their relationships"""
        
        for book in books:
            # Prepare book data
            book_data = {
                'book_id': book.get('book_id', f"book_{hash(book.get('title', ''))}"),
                'title': book.get('title', ''),
                'isbn': book.get('isbn', ''),
                'year_published': book.get('year_published', 0),
                'pages': book.get('pages', 0),
                'average_rating': book.get('average_rating', 0.0),
                'ratings_count': book.get('ratings_count', 0),
                'description': book.get('description', ''),
                'source': book.get('source', ''),
                'google_id': book.get('google_id', ''),
                'goodreads_id': book.get('goodreads_id', ''),
                'popularity_score': book.get('popularity_score', 0.0),
                'recency_score': book.get('recency_score', 0.0),
                'recommendation_score': book.get('recommendation_score', 0.0),
                'length_category': book.get('length_category', ''),
                'popularity_tier': book.get('popularity_tier', ''),
                'tag': 'bookrecommender',  # Important: tag to distinguish from Xerox docs
                'created_at': datetime.now().isoformat(),
                'embedding': book.get('embedding', [])
            }
            
            # Create book node
            book_query = """
            MERGE (b:Book {book_id: $book_id, tag: $tag})
            SET b += $book_data
            """
            tx.run(book_query, book_id=book_data['book_id'], tag='bookrecommender', book_data=book_data)
            
            # Create author and relationship
            primary_author = book.get('primary_author', 'Unknown')
            if primary_author and primary_author != 'Unknown':
                author_query = """
                MERGE (a:Author {name: $author_name})
                SET a.tag = $tag,
                    a.created_at = CASE WHEN a.created_at IS NULL THEN $created_at ELSE a.created_at END
                WITH a
                MATCH (b:Book {book_id: $book_id, tag: $tag})
                MERGE (a)-[:WROTE]->(b)
                """
                tx.run(author_query, 
                      author_name=primary_author, 
                      tag='bookrecommender',
                      created_at=datetime.now().isoformat(),
                      book_id=book_data['book_id'])
            
            # Create genres and relationships
            genres = book.get('normalized_genres', [])
            for genre in genres:
                if genre:
                    genre_query = """
                    MERGE (g:Genre {name: $genre_name, tag: $tag})
                    SET g.created_at = CASE WHEN g.created_at IS NULL THEN $created_at ELSE g.created_at END
                    WITH g
                    MATCH (b:Book {book_id: $book_id, tag: $tag})
                    MERGE (b)-[:BELONGS_TO]->(g)
                    """
                    tx.run(genre_query, 
                          genre_name=genre, 
                          tag='bookrecommender',
                          created_at=datetime.now().isoformat(),
                          book_id=book_data['book_id'])
            
            # Create publisher and relationship
            publisher = book.get('publisher', '')
            if publisher and publisher.strip():
                publisher_query = """
                MERGE (p:Publisher {name: $publisher_name, tag: $tag})
                SET p.created_at = CASE WHEN p.created_at IS NULL THEN $created_at ELSE p.created_at END
                WITH p
                MATCH (b:Book {book_id: $book_id, tag: $tag})
                MERGE (p)-[:PUBLISHED]->(b)
                """
                tx.run(publisher_query, 
                      publisher_name=publisher.strip(), 
                      tag='bookrecommender',
                      created_at=datetime.now().isoformat(),
                      book_id=book_data['book_id'])
    
    def create_similarity_relationships(self, similarity_threshold: float = 0.8, max_relationships: int = 1000):
        """Create SIMILAR_TO relationships between books based on embedding similarity"""
        logger.info(f"Creating similarity relationships (threshold: {similarity_threshold})...")
        
        with self.driver.session() as session:
            # Query to find similar books using vector similarity
            similarity_query = """
            MATCH (b1:Book {tag: 'bookrecommender'})
            WHERE b1.embedding IS NOT NULL
            MATCH (b2:Book {tag: 'bookrecommender'})
            WHERE b2.embedding IS NOT NULL AND b1.book_id < b2.book_id
            WITH b1, b2, 
                 gds.similarity.cosine(b1.embedding, b2.embedding) AS similarity
            WHERE similarity >= $threshold
            WITH b1, b2, similarity
            ORDER BY similarity DESC
            LIMIT $max_relationships
            MERGE (b1)-[r:SIMILAR_TO]-(b2)
            SET r.similarity = similarity, r.created_at = datetime()
            RETURN count(r) as relationships_created
            """
            
            try:
                result = session.run(similarity_query, 
                                   threshold=similarity_threshold, 
                                   max_relationships=max_relationships)
                relationships_created = result.single()["relationships_created"]
                logger.info(f"✅ Created {relationships_created} similarity relationships")
            except Exception as e:
                logger.warning(f"⚠️  Could not create similarity relationships: {e}")
                logger.info("This might be due to missing GDS library or vector index not ready")
    
    def get_upload_summary(self) -> Dict[str, Any]:
        """Get summary of uploaded data"""
        logger.info("Generating upload summary...")
        
        with self.driver.session() as session:
            summary_queries = {
                'total_books': "MATCH (b:Book {tag: 'bookrecommender'}) RETURN count(b) as count",
                'total_authors': "MATCH (a:Author {tag: 'bookrecommender'}) RETURN count(a) as count",
                'total_genres': "MATCH (g:Genre {tag: 'bookrecommender'}) RETURN count(g) as count",
                'total_publishers': "MATCH (p:Publisher {tag: 'bookrecommender'}) RETURN count(p) as count",
                'wrote_relationships': "MATCH (:Author {tag: 'bookrecommender'})-[r:WROTE]->(:Book {tag: 'bookrecommender'}) RETURN count(r) as count",
                'belongs_to_relationships': "MATCH (:Book {tag: 'bookrecommender'})-[r:BELONGS_TO]->(:Genre {tag: 'bookrecommender'}) RETURN count(r) as count",
                'published_relationships': "MATCH (:Publisher {tag: 'bookrecommender'})-[r:PUBLISHED]->(:Book {tag: 'bookrecommender'}) RETURN count(r) as count",
                'similarity_relationships': "MATCH (:Book {tag: 'bookrecommender'})-[r:SIMILAR_TO]-(:Book {tag: 'bookrecommender'}) RETURN count(r) as count"
            }
            
            summary = {}
            for key, query in summary_queries.items():
                try:
                    result = session.run(query)
                    summary[key] = result.single()["count"]
                except Exception as e:
                    logger.warning(f"Could not get {key}: {e}")
                    summary[key] = 0
            
            # Get sample books
            try:
                sample_query = """
                MATCH (b:Book {tag: 'bookrecommender'})
                RETURN b.title as title, b.primary_author as author, b.normalized_genres as genres
                LIMIT 5
                """
                result = session.run(sample_query)
                summary['sample_books'] = [dict(record) for record in result]
            except Exception as e:
                logger.warning(f"Could not get sample books: {e}")
                summary['sample_books'] = []
        
        return summary
    
    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")

def load_books_with_embeddings() -> List[Dict[str, Any]]:
    """Load books with embeddings"""
    try:
        with open('books_with_embeddings.json', 'r', encoding='utf-8') as f:
            books = json.load(f)
        logger.info(f"Loaded {len(books)} books with embeddings")
        return books
    except FileNotFoundError:
        logger.error("books_with_embeddings.json not found. Please run vectorize_books.py first.")
        return []

def main():
    """Main upload process"""
    logger.info("=== Neo4j Book Upload Process ===")
    
    if not NEO4J_AVAILABLE:
        logger.error("Neo4j driver not available. Install with: pip install neo4j")
        return
    
    # Load books with embeddings
    books = load_books_with_embeddings()
    if not books:
        return
    
    try:
        # Initialize Neo4j uploader
        uploader = Neo4jBookUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        # Check for existing Xerox documents
        logger.info("\n=== Checking for Xerox Documents ===")
        xerox_info = uploader.check_xerox_documents()
        
        logger.info("Xerox Document Analysis:")
        for key, value in xerox_info.items():
            if key != 'sample_nodes':
                logger.info(f"  {key}: {value}")
        
        if xerox_info.get('total_nodes', 0) > 0:
            logger.info(f"✅ Found {xerox_info['total_nodes']} existing nodes in database")
            if any(count > 0 for key, count in xerox_info.items() 
                   if key.startswith('xerox') and isinstance(count, int)):
                logger.info("✅ Xerox-related documents detected and will be preserved")
            else:
                logger.info("ℹ️  No explicit Xerox documents found, but existing data will be preserved")
        else:
            logger.info("ℹ️  Database appears to be empty")
        
        # Create indexes and constraints
        logger.info("\n=== Setting up Database Schema ===")
        uploader.create_indexes_and_constraints()
        
        # Upload books
        logger.info("\n=== Uploading Books ===")
        uploader.upload_books_batch(books)
        
        # Create similarity relationships
        logger.info("\n=== Creating Similarity Relationships ===")
        uploader.create_similarity_relationships()
        
        # Get upload summary
        logger.info("\n=== Upload Summary ===")
        summary = uploader.get_upload_summary()
        
        logger.info("Upload completed successfully!")
        logger.info(f"  📚 Books uploaded: {summary['total_books']}")
        logger.info(f"  👥 Authors created: {summary['total_authors']}")
        logger.info(f"  🏷️  Genres created: {summary['total_genres']}")
        logger.info(f"  🏢 Publishers created: {summary['total_publishers']}")
        logger.info(f"  🔗 WROTE relationships: {summary['wrote_relationships']}")
        logger.info(f"  🔗 BELONGS_TO relationships: {summary['belongs_to_relationships']}")
        logger.info(f"  🔗 PUBLISHED relationships: {summary['published_relationships']}")
        logger.info(f"  🔗 SIMILAR_TO relationships: {summary['similarity_relationships']}")
        
        if summary['sample_books']:
            logger.info("\nSample uploaded books:")
            for i, book in enumerate(summary['sample_books'][:3]):
                logger.info(f"  {i+1}. '{book['title']}' by {book.get('author', 'Unknown')}")
        
        # Save summary
        upload_summary = {
            'upload_timestamp': datetime.now().isoformat(),
            'books_uploaded': len(books),
            'neo4j_summary': summary,
            'xerox_analysis': xerox_info,
            'tag_used': 'bookrecommender'
        }
        
        with open('neo4j_upload_summary.json', 'w', encoding='utf-8') as f:
            json.dump(upload_summary, f, indent=2, ensure_ascii=False)
        
        logger.info("\nFiles created:")
        logger.info("  - neo4j_upload_summary.json")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise
    finally:
        if 'uploader' in locals():
            uploader.close()

if __name__ == "__main__":
    main()