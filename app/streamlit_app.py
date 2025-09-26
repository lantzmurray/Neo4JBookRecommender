#!/usr/bin/env python3
"""
Book Recommendation System - Streamlit Web Application

This application provides a web interface for the book recommendation system,
allowing users to search for books and get personalized recommendations based on
content similarity using Neo4j graph database and OpenAI embeddings.

Features:
- Book search with filters (genre, rating, publication year)
- Content-based recommendations using cosine similarity
- Interactive web interface with Streamlit
- Real-time database queries to Neo4j
- Responsive design with custom CSS styling

Author: Book Recommendation System Team
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import openai
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .book-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .book-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .book-author {
        color: #7f8c8d;
        font-style: italic;
        margin-bottom: 0.5rem;
    }
    .book-rating {
        color: #f39c12;
        font-weight: bold;
    }
    .similarity-score {
        background-color: #e8f5e8;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        color: #27ae60;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

class BookRecommendationSystem:
    """Main class for the book recommendation system."""
    
    def __init__(self):
        """Initialize the recommendation system with database connections."""
        self.driver = None
        self.openai_client = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize Neo4j and OpenAI connections."""
        try:
            # Initialize Neo4j connection
            self.initialize_neo4j()
            
            # Initialize OpenAI client
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                openai.api_key = openai_api_key
                self.openai_client = openai
                logger.info("OpenAI client initialized successfully")
            else:
                logger.warning("OpenAI API key not found")
                
        except Exception as e:
            logger.error(f"Error initializing connections: {e}")
            st.error(f"Failed to initialize connections: {e}")
    
    def initialize_neo4j(self):
        """Initialize Neo4j database connection."""
        try:
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
            
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            logger.info("Neo4j connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            st.error(f"Database connection failed: {e}")
            raise
    
    def load_books_data(self, limit: int = 1000) -> List[Dict]:
        """Load books data from Neo4j database."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (b:Book)
                OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
                OPTIONAL MATCH (b)-[:BELONGS_TO]->(g:Genre)
                RETURN b.title as title,
                       b.author as author,
                       b.rating as rating,
                       b.publication_year as publication_year,
                       b.description as description,
                       b.isbn as isbn,
                       b.page_count as page_count,
                       b.language as language,
                       collect(DISTINCT a.name) as authors,
                       collect(DISTINCT g.name) as genres
                LIMIT $limit
                """
                
                result = session.run(query, limit=limit)
                books = []
                
                for record in result:
                    book = dict(record)
                    # Clean up the data
                    book['authors'] = [a for a in book['authors'] if a] or [book['author']]
                    book['genres'] = [g for g in book['genres'] if g] or ['Unknown']
                    books.append(book)
                
                logger.info(f"Loaded {len(books)} books from database")
                return books
                
        except Exception as e:
            logger.error(f"Error loading books data: {e}")
            st.error(f"Failed to load books: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_similar_books(self, book_title: str, limit: int = 10) -> List[Dict]:
        """Get similar books based on embeddings."""
        try:
            with self.driver.session() as session:
                # First, get the target book's embedding
                query = """
                MATCH (b:Book {title: $title})
                RETURN b.embedding as embedding, b.title as title
                """
                
                result = session.run(query, title=book_title)
                target_record = result.single()
                
                if not target_record or not target_record['embedding']:
                    return []
                
                target_embedding = target_record['embedding']
                
                # Get all books with embeddings
                query = """
                MATCH (b:Book)
                WHERE b.embedding IS NOT NULL AND b.title <> $title
                OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
                OPTIONAL MATCH (b)-[:BELONGS_TO]->(g:Genre)
                RETURN b.title as title,
                       b.author as author,
                       b.rating as rating,
                       b.publication_year as publication_year,
                       b.description as description,
                       b.embedding as embedding,
                       collect(DISTINCT a.name) as authors,
                       collect(DISTINCT g.name) as genres
                LIMIT 1000
                """
                
                result = session.run(query, title=book_title)
                similar_books = []
                
                for record in result:
                    book = dict(record)
                    if book['embedding']:
                        similarity = self.cosine_similarity(target_embedding, book['embedding'])
                        book['similarity'] = similarity
                        book['authors'] = [a for a in book['authors'] if a] or [book['author']]
                        book['genres'] = [g for g in book['genres'] if g] or ['Unknown']
                        similar_books.append(book)
                
                # Sort by similarity and return top results
                similar_books.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_books[:limit]
                
        except Exception as e:
            logger.error(f"Error getting similar books: {e}")
            st.error(f"Failed to get recommendations: {e}")
            return []
    
    def search_books(self, 
                    title_query: str = "", 
                    author_query: str = "", 
                    genre_filter: str = "All",
                    min_rating: float = 0.0,
                    min_year: int = 1900,
                    max_year: int = 2024,
                    limit: int = 50) -> List[Dict]:
        """Search books with various filters."""
        try:
            with self.driver.session() as session:
                # Build dynamic query
                conditions = []
                params = {'limit': limit}
                
                if title_query:
                    conditions.append("toLower(b.title) CONTAINS toLower($title)")
                    params['title'] = title_query
                
                if author_query:
                    conditions.append("toLower(b.author) CONTAINS toLower($author)")
                    params['author'] = author_query
                
                if genre_filter != "All":
                    conditions.append("ANY(genre IN b.genres WHERE toLower(genre) CONTAINS toLower($genre))")
                    params['genre'] = genre_filter
                
                if min_rating > 0:
                    conditions.append("b.rating >= $min_rating")
                    params['min_rating'] = min_rating
                
                if min_year > 1900:
                    conditions.append("b.publication_year >= $min_year")
                    params['min_year'] = min_year
                
                if max_year < 2024:
                    conditions.append("b.publication_year <= $max_year")
                    params['max_year'] = max_year
                
                where_clause = " AND ".join(conditions) if conditions else "1=1"
                
                query = f"""
                MATCH (b:Book)
                WHERE {where_clause}
                OPTIONAL MATCH (b)-[:WRITTEN_BY]->(a:Author)
                OPTIONAL MATCH (b)-[:BELONGS_TO]->(g:Genre)
                RETURN b.title as title,
                       b.author as author,
                       b.rating as rating,
                       b.publication_year as publication_year,
                       b.description as description,
                       b.isbn as isbn,
                       b.page_count as page_count,
                       b.language as language,
                       collect(DISTINCT a.name) as authors,
                       collect(DISTINCT g.name) as genres
                ORDER BY b.rating DESC, b.title ASC
                LIMIT $limit
                """
                
                result = session.run(query, **params)
                books = []
                
                for record in result:
                    book = dict(record)
                    book['authors'] = [a for a in book['authors'] if a] or [book['author']]
                    book['genres'] = [g for g in book['genres'] if g] or ['Unknown']
                    books.append(book)
                
                return books
                
        except Exception as e:
            logger.error(f"Error searching books: {e}")
            st.error(f"Search failed: {e}")
            return []
    
    def get_available_genres(self) -> List[str]:
        """Get list of available genres from the database."""
        try:
            with self.driver.session() as session:
                query = """
                MATCH (g:Genre)
                RETURN DISTINCT g.name as genre
                ORDER BY g.name
                """
                
                result = session.run(query)
                genres = [record['genre'] for record in result if record['genre']]
                
                # Also get genres from book properties
                query = """
                MATCH (b:Book)
                WHERE b.genres IS NOT NULL
                UNWIND b.genres as genre
                RETURN DISTINCT genre
                ORDER BY genre
                """
                
                result = session.run(query)
                book_genres = [record['genre'] for record in result if record['genre']]
                
                # Combine and deduplicate
                all_genres = list(set(genres + book_genres))
                all_genres.sort()
                
                return all_genres
                
        except Exception as e:
            logger.error(f"Error getting genres: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        try:
            with self.driver.session() as session:
                stats = {}
                
                # Count books
                result = session.run("MATCH (b:Book) RETURN count(b) as count")
                stats['books'] = result.single()['count']
                
                # Count authors
                result = session.run("MATCH (a:Author) RETURN count(a) as count")
                stats['authors'] = result.single()['count']
                
                # Count genres
                result = session.run("MATCH (g:Genre) RETURN count(g) as count")
                stats['genres'] = result.single()['count']
                
                # Count books with embeddings
                result = session.run("MATCH (b:Book) WHERE b.embedding IS NOT NULL RETURN count(b) as count")
                stats['books_with_embeddings'] = result.single()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

def display_book_card(book: Dict, show_similarity: bool = False):
    """Display a book in a card format."""
    with st.container():
        st.markdown('<div class="book-card">', unsafe_allow_html=True)
        
        # Title and similarity score
        title_html = f'<div class="book-title">{book["title"]}</div>'
        if show_similarity and 'similarity' in book:
            similarity_html = f'<span class="similarity-score">Similarity: {book["similarity"]:.3f}</span>'
            title_html += similarity_html
        st.markdown(title_html, unsafe_allow_html=True)
        
        # Author
        authors = book.get('authors', [book.get('author', 'Unknown')])
        author_text = ', '.join(authors) if isinstance(authors, list) else str(authors)
        st.markdown(f'<div class="book-author">by {author_text}</div>', unsafe_allow_html=True)
        
        # Rating and year
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            rating = book.get('rating', 0)
            if rating:
                st.markdown(f'<div class="book-rating">⭐ {rating:.1f}</div>', unsafe_allow_html=True)
        
        with col2:
            year = book.get('publication_year')
            if year:
                st.write(f"📅 {year}")
        
        with col3:
            genres = book.get('genres', ['Unknown'])
            if isinstance(genres, list) and genres:
                genre_text = ', '.join(genres[:3])  # Show max 3 genres
                st.write(f"📚 {genre_text}")
        
        # Description
        description = book.get('description', '')
        if description:
            # Truncate long descriptions
            if len(description) > 200:
                description = description[:200] + "..."
            st.write(description)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Book Recommendation System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main header
    st.markdown('<h1 class="main-header">📚 Book Recommendation System</h1>', unsafe_allow_html=True)
    
    # Initialize the recommendation system
    if 'recommender' not in st.session_state:
        with st.spinner("Initializing recommendation system..."):
            st.session_state.recommender = BookRecommendationSystem()
    
    recommender = st.session_state.recommender
    
    # Sidebar for navigation and filters
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🔍 Search Books", "💡 Get Recommendations", "📊 Database Stats"]
    )
    
    if page == "🔍 Search Books":
        st.header("Search Books")
        
        # Search filters
        col1, col2 = st.columns(2)
        
        with col1:
            title_query = st.text_input("Search by title:", placeholder="Enter book title...")
            author_query = st.text_input("Search by author:", placeholder="Enter author name...")
        
        with col2:
            # Get available genres
            genres = ["All"] + recommender.get_available_genres()
            genre_filter = st.selectbox("Filter by genre:", genres)
            
            min_rating = st.slider("Minimum rating:", 0.0, 5.0, 0.0, 0.1)
        
        # Year range
        col3, col4 = st.columns(2)
        with col3:
            min_year = st.number_input("From year:", min_value=1900, max_value=2024, value=1900)
        with col4:
            max_year = st.number_input("To year:", min_value=1900, max_value=2024, value=2024)
        
        # Search button
        if st.button("🔍 Search Books", type="primary"):
            with st.spinner("Searching books..."):
                books = recommender.search_books(
                    title_query=title_query,
                    author_query=author_query,
                    genre_filter=genre_filter,
                    min_rating=min_rating,
                    min_year=min_year,
                    max_year=max_year,
                    limit=50
                )
            
            if books:
                st.success(f"Found {len(books)} books matching your criteria!")
                
                # Display results
                for book in books:
                    display_book_card(book)
            else:
                st.warning("No books found matching your criteria. Try adjusting your filters.")
    
    elif page == "💡 Get Recommendations":
        st.header("Get Book Recommendations")
        
        # Book selection for recommendations
        st.write("Enter a book title to get similar recommendations:")
        
        # Load some popular books for suggestions
        with st.spinner("Loading book suggestions..."):
            sample_books = recommender.load_books_data(limit=100)
        
        if sample_books:
            book_titles = [book['title'] for book in sample_books]
            
            # Selectbox with search
            selected_title = st.selectbox(
                "Choose a book or type to search:",
                options=[""] + book_titles,
                format_func=lambda x: "Select a book..." if x == "" else x
            )
            
            # Alternative text input
            st.write("Or enter a book title manually:")
            manual_title = st.text_input("Book title:", placeholder="Enter exact book title...")
            
            # Use manual input if provided, otherwise use selection
            target_title = manual_title if manual_title else selected_title
            
            if target_title and st.button("🎯 Get Recommendations", type="primary"):
                with st.spinner(f"Finding books similar to '{target_title}'..."):
                    recommendations = recommender.get_similar_books(target_title, limit=10)
                
                if recommendations:
                    st.success(f"Found {len(recommendations)} similar books!")
                    
                    # Display recommendations
                    for book in recommendations:
                        display_book_card(book, show_similarity=True)
                else:
                    st.warning(f"No recommendations found for '{target_title}'. Please check the title or try another book.")
        else:
            st.error("Unable to load book data for recommendations.")
    
    elif page == "📊 Database Stats":
        st.header("Database Statistics")
        
        with st.spinner("Loading database statistics..."):
            stats = recommender.get_database_stats()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Books", stats.get('books', 0))
            
            with col2:
                st.metric("Total Authors", stats.get('authors', 0))
            
            with col3:
                st.metric("Total Genres", stats.get('genres', 0))
            
            with col4:
                st.metric("Books with Embeddings", stats.get('books_with_embeddings', 0))
            
            # Additional info
            st.subheader("System Information")
            st.write(f"**Database Connection:** ✅ Connected")
            st.write(f"**OpenAI Integration:** {'✅ Active' if recommender.openai_client else '❌ Not configured'}")
            st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.error("Unable to load database statistics.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Book Recommendation System | Powered by Neo4j, OpenAI, and Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()