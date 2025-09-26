#!/usr/bin/env python3
"""
Book Vectorization and Embedding Script
Creates vector embeddings for books using OpenAI's text-embedding-ada-002 model
"""

import json
import pandas as pd
import numpy as np
import openai
import os
import time
from datetime import datetime
import tiktoken
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. Please set it to use OpenAI embeddings.")

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIMENSION = 1536
MAX_TOKENS = 8191  # Max tokens for ada-002

def load_enriched_data():
    """Load the merged book dataset"""
    try:
        # Try merged dataset first
        with open('merged_book_dataset.json', 'r', encoding='utf-8') as f:
            books = json.load(f)
        logger.info(f"Loaded {len(books)} books from merged dataset")
        return books
    except FileNotFoundError:
        try:
            # Fallback to enriched dataset
            with open('enriched_book_dataset.json', 'r', encoding='utf-8') as f:
                books = json.load(f)
            logger.info(f"Loaded {len(books)} books from enriched dataset")
            return books
        except FileNotFoundError:
            logger.error("Neither merged_book_dataset.json nor enriched_book_dataset.json found.")
            return []

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(EMBEDDING_MODEL)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        return len(text.split()) * 1.3  # Rough estimate

def create_book_text_representation(book: Dict[str, Any]) -> str:
    """Create a comprehensive text representation of a book for embedding"""
    
    # Core information
    title = book.get('title', '').strip()
    author = book.get('primary_author', '').strip()
    description = book.get('description', '').strip()
    
    # Metadata
    genres = book.get('normalized_genres', [])
    year = book.get('year_published', '')
    publisher = book.get('publisher', '').strip()
    
    # Additional features
    keywords = book.get('description_keywords', [])
    categories = book.get('categories', [])
    
    # Build comprehensive text representation
    text_parts = []
    
    # Title and author (most important)
    if title:
        text_parts.append(f"Title: {title}")
    if author and author != 'Unknown':
        text_parts.append(f"Author: {author}")
    
    # Genres and categories
    if genres:
        text_parts.append(f"Genres: {', '.join(genres)}")
    if categories:
        text_parts.append(f"Categories: {', '.join(categories)}")
    
    # Publication info
    if year:
        text_parts.append(f"Published: {year}")
    if publisher:
        text_parts.append(f"Publisher: {publisher}")
    
    # Keywords from description
    if keywords:
        text_parts.append(f"Keywords: {', '.join(keywords[:10])}")  # Limit to top 10
    
    # Description (truncated if too long)
    if description:
        # Truncate description to fit within token limits
        max_desc_tokens = MAX_TOKENS - count_tokens(' '.join(text_parts)) - 100  # Buffer
        if max_desc_tokens > 100:
            desc_tokens = count_tokens(description)
            if desc_tokens > max_desc_tokens:
                # Truncate description
                words = description.split()
                truncated_words = words[:int(len(words) * max_desc_tokens / desc_tokens)]
                description = ' '.join(truncated_words) + '...'
            text_parts.append(f"Description: {description}")
    
    return ' | '.join(text_parts)

def get_embedding_batch(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Get embeddings for a batch of texts using OpenAI API"""
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key not available. Cannot generate embeddings.")
        return []
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            # Rate limiting - be respectful to API
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error getting embeddings for batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed batch
            embeddings.extend([[0.0] * EMBEDDING_DIMENSION] * len(batch))
    
    return embeddings

def create_fallback_embeddings(books: List[Dict[str, Any]]) -> List[List[float]]:
    """Create simple fallback embeddings based on book features when OpenAI is not available"""
    logger.info("Creating fallback embeddings based on book features...")
    
    embeddings = []
    
    # Get all unique genres and authors for feature encoding
    all_genres = set()
    all_authors = set()
    
    for book in books:
        all_genres.update(book.get('normalized_genres', []))
        all_authors.add(book.get('primary_author', 'Unknown'))
    
    genre_list = sorted(list(all_genres))
    author_list = sorted(list(all_authors))
    
    logger.info(f"Creating embeddings with {len(genre_list)} genres and {len(author_list)} authors")
    
    for book in books:
        # Create feature vector
        features = []
        
        # Genre features (one-hot encoding for top genres)
        genre_features = [1 if genre in book.get('normalized_genres', []) else 0 
                         for genre in genre_list[:50]]  # Limit to top 50 genres
        features.extend(genre_features)
        
        # Numerical features (normalized)
        features.append(book.get('average_rating', 0) / 5.0)  # Normalize rating
        features.append(min(book.get('ratings_count', 0) / 10000.0, 1.0))  # Normalize ratings count
        features.append(book.get('recency_score', 0) / 10.0)  # Normalize recency
        features.append(book.get('recommendation_score', 0) / 10.0)  # Normalize rec score
        features.append(min(book.get('pages', 0) / 1000.0, 1.0))  # Normalize pages
        
        # Year feature (normalized to 0-1 range for recent years)
        year = book.get('year_published', 2020)
        if isinstance(year, str):
            try:
                year = int(year)
            except ValueError:
                year = 2020
        features.append((year - 2000) / 25.0)  # Normalize years 2000-2025
        
        # Popularity tier encoding
        popularity_tier = book.get('popularity_tier', 'low')
        features.extend([
            1 if popularity_tier == 'high' else 0,
            1 if popularity_tier == 'medium' else 0,
            1 if popularity_tier == 'low' else 0
        ])
        
        # Length category encoding
        length_cat = book.get('length_category', 'medium')
        features.extend([
            1 if length_cat == 'short' else 0,
            1 if length_cat == 'medium' else 0,
            1 if length_cat == 'long' else 0
        ])
        
        # Pad or truncate to desired dimension
        if len(features) < EMBEDDING_DIMENSION:
            features.extend([0.0] * (EMBEDDING_DIMENSION - len(features)))
        else:
            features = features[:EMBEDDING_DIMENSION]
        
        embeddings.append(features)
    
    return embeddings

def calculate_similarity_matrix(embeddings: List[List[float]], sample_size: int = 100) -> np.ndarray:
    """Calculate cosine similarity matrix for a sample of embeddings"""
    logger.info(f"Calculating similarity matrix for {min(sample_size, len(embeddings))} books...")
    
    # Take a sample for similarity calculation (to avoid memory issues)
    sample_embeddings = embeddings[:sample_size]
    embeddings_array = np.array(sample_embeddings)
    
    # Calculate cosine similarity
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    normalized_embeddings = embeddings_array / (norms + 1e-8)  # Avoid division by zero
    
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    return similarity_matrix

def save_embeddings_and_analysis(books: List[Dict[str, Any]], embeddings: List[List[float]]):
    """Save embeddings and perform analysis"""
    logger.info("Saving embeddings and performing analysis...")
    
    # Add embeddings to books
    for book, embedding in zip(books, embeddings):
        book['embedding'] = embedding
    
    # Save books with embeddings
    with open('books_with_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(books, f, indent=2, ensure_ascii=False)
    
    # Save embeddings separately (for easier loading)
    embeddings_data = {
        'embeddings': embeddings,
        'model': EMBEDDING_MODEL,
        'dimension': EMBEDDING_DIMENSION,
        'book_count': len(books),
        'created_at': datetime.now().isoformat()
    }
    
    with open('book_embeddings.json', 'w', encoding='utf-8') as f:
        json.dump(embeddings_data, f, indent=2)
    
    # Calculate and save similarity analysis
    similarity_matrix = calculate_similarity_matrix(embeddings, sample_size=100)
    
    # Find most similar book pairs
    similar_pairs = []
    n_sample = min(100, len(books))
    
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            similarity = similarity_matrix[i, j]
            if similarity > 0.7:  # High similarity threshold
                similar_pairs.append({
                    'book1': {
                        'title': books[i].get('title', ''),
                        'author': books[i].get('primary_author', ''),
                        'genres': books[i].get('normalized_genres', [])
                    },
                    'book2': {
                        'title': books[j].get('title', ''),
                        'author': books[j].get('primary_author', ''),
                        'genres': books[j].get('normalized_genres', [])
                    },
                    'similarity': float(similarity)
                })
    
    # Sort by similarity
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Create analysis summary
    analysis = {
        'total_books': len(books),
        'embedding_dimension': EMBEDDING_DIMENSION,
        'model_used': EMBEDDING_MODEL if OPENAI_API_KEY else 'fallback_features',
        'high_similarity_pairs': len(similar_pairs),
        'top_similar_pairs': similar_pairs[:20],  # Top 20 most similar pairs
        'embedding_stats': {
            'mean_embedding_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings[:100]])),
            'std_embedding_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings[:100]])),
        },
        'similarity_stats': {
            'mean_similarity': float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
            'max_similarity': float(np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])),
            'min_similarity': float(np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        },
        'created_at': datetime.now().isoformat()
    }
    
    with open('embedding_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    return analysis

def main():
    """Main vectorization process"""
    logger.info("=== Book Vectorization and Embedding Process ===")
    
    # Load enriched data
    books = load_enriched_data()
    if not books:
        return
    
    logger.info(f"Processing {len(books)} books for embedding...")
    
    # Create text representations
    logger.info("Creating text representations for embedding...")
    text_representations = []
    
    for i, book in enumerate(books):
        if i % 500 == 0:
            logger.info(f"Processing book {i+1}/{len(books)}")
        
        text_repr = create_book_text_representation(book)
        text_representations.append(text_repr)
        
        # Check token count
        token_count = count_tokens(text_repr)
        if token_count > MAX_TOKENS:
            logger.warning(f"Book {i+1} text representation has {token_count} tokens (max: {MAX_TOKENS})")
    
    # Generate embeddings
    if OPENAI_API_KEY:
        logger.info("Generating embeddings using OpenAI API...")
        embeddings = get_embedding_batch(text_representations)
    else:
        logger.info("OpenAI API key not available. Using fallback feature-based embeddings...")
        embeddings = create_fallback_embeddings(books)
    
    if not embeddings:
        logger.error("Failed to generate embeddings")
        return
    
    # Save embeddings and perform analysis
    analysis = save_embeddings_and_analysis(books, embeddings)
    
    logger.info("\n=== Vectorization Complete ===")
    logger.info(f"Total books processed: {analysis['total_books']}")
    logger.info(f"Embedding dimension: {analysis['embedding_dimension']}")
    logger.info(f"Model used: {analysis['model_used']}")
    logger.info(f"High similarity pairs found: {analysis['high_similarity_pairs']}")
    logger.info(f"Mean similarity score: {analysis['similarity_stats']['mean_similarity']:.3f}")
    
    if analysis['top_similar_pairs']:
        logger.info("\nTop 3 most similar book pairs:")
        for i, pair in enumerate(analysis['top_similar_pairs'][:3]):
            logger.info(f"  {i+1}. '{pair['book1']['title']}' by {pair['book1']['author']}")
            logger.info(f"     '{pair['book2']['title']}' by {pair['book2']['author']}")
            logger.info(f"     Similarity: {pair['similarity']:.3f}")
    
    logger.info("\nFiles created:")
    logger.info("  - books_with_embeddings.json")
    logger.info("  - book_embeddings.json")
    logger.info("  - embedding_analysis.json")

if __name__ == "__main__":
    main()