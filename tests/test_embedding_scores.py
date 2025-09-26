#!/usr/bin/env python3
"""
Embedding Score Testing Script
Tests embedding quality using most recent Goodreads books from user data
Validates recommendation system performance
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def load_original_goodreads_data() -> List[Dict[str, Any]]:
    """Load original Goodreads data to identify user's recent reads"""
    try:
        with open('goodreads_analyzed_books.json', 'r', encoding='utf-8') as f:
            goodreads_books = json.load(f)
        logger.info(f"Loaded {len(goodreads_books)} original Goodreads books")
        return goodreads_books
    except FileNotFoundError:
        logger.error("goodreads_analyzed_books.json not found.")
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def parse_date(date_str: str) -> datetime:
    """Parse various date formats"""
    if not date_str or date_str.strip() == '':
        return datetime(2020, 1, 1)  # Default date
    
    # Common date formats
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y',
        '%B %d, %Y',
        '%b %d, %Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Try to extract year from string
    year_match = re.search(r'\b(20\d{2})\b', date_str)
    if year_match:
        return datetime(int(year_match.group(1)), 1, 1)
    
    return datetime(2020, 1, 1)  # Default fallback

def get_recent_user_books(goodreads_books: List[Dict[str, Any]], months_back: int = 12) -> List[Dict[str, Any]]:
    """Get user's most recent reads from Goodreads data"""
    logger.info(f"Finding books read in the last {months_back} months...")
    
    cutoff_date = datetime.now() - timedelta(days=months_back * 30)
    recent_books = []
    
    for book in goodreads_books:
        # Check for read status and date_read field
        read_status = book.get('read_status', '')
        date_read_str = book.get('date_read', '')
        
        # Only consider books that have been read (not currently-reading or to-read)
        if read_status == 'read' and date_read_str and str(date_read_str) != 'NaN':
            read_date = parse_date(str(date_read_str))
            if read_date >= cutoff_date:
                recent_books.append({
                    **book,
                    'parsed_read_date': read_date
                })
    
    # Sort by read date (most recent first)
    recent_books.sort(key=lambda x: x['parsed_read_date'], reverse=True)
    
    logger.info(f"Found {len(recent_books)} books read in the last {months_back} months")
    
    if recent_books:
        logger.info("Most recent reads:")
        for i, book in enumerate(recent_books[:5]):
            logger.info(f"  {i+1}. '{book.get('title', 'Unknown')}' by {book.get('author', 'Unknown')} "
                       f"(read: {book['parsed_read_date'].strftime('%Y-%m-%d')})")
    
    return recent_books

def find_book_in_embeddings(target_book: Dict[str, Any], books_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find a book in the embeddings dataset"""
    target_title = target_book.get('title', '').lower().strip()
    target_author = target_book.get('author', '').lower().strip()
    
    # Try exact title match first
    for book in books_with_embeddings:
        book_title = book.get('title', '').lower().strip()
        book_author = book.get('primary_author', '').lower().strip()
        
        if book_title == target_title and book_author == target_author:
            return book
    
    # Try partial title match
    for book in books_with_embeddings:
        book_title = book.get('title', '').lower().strip()
        book_author = book.get('primary_author', '').lower().strip()
        
        if (target_title in book_title or book_title in target_title) and \
           (target_author in book_author or book_author in target_author):
            return book
    
    return None

def test_content_based_recommendations(recent_books: List[Dict[str, Any]], 
                                     books_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test content-based recommendations using embeddings"""
    logger.info("Testing content-based recommendations...")
    
    test_results = {
        'total_recent_books': len(recent_books),
        'books_found_in_embeddings': 0,
        'recommendation_tests': [],
        'average_similarity_scores': [],
        'genre_match_accuracy': [],
        'author_match_accuracy': []
    }
    
    for recent_book in recent_books[:10]:  # Test with up to 10 recent books
        # Find the book in our embeddings dataset
        found_book = find_book_in_embeddings(recent_book, books_with_embeddings)
        
        if not found_book or not found_book.get('embedding'):
            continue
        
        test_results['books_found_in_embeddings'] += 1
        
        # Get recommendations based on embedding similarity
        recommendations = get_similar_books(found_book, books_with_embeddings, top_k=10)
        
        # Analyze recommendation quality
        test_result = analyze_recommendation_quality(recent_book, found_book, recommendations)
        test_results['recommendation_tests'].append(test_result)
        
        # Collect metrics
        if test_result['similarities']:
            test_results['average_similarity_scores'].extend(test_result['similarities'])
        
        test_results['genre_match_accuracy'].append(test_result['genre_match_rate'])
        test_results['author_match_accuracy'].append(test_result['author_match_rate'])
    
    # Calculate overall metrics
    if test_results['average_similarity_scores']:
        test_results['mean_similarity'] = np.mean(test_results['average_similarity_scores'])
        test_results['std_similarity'] = np.std(test_results['average_similarity_scores'])
    else:
        test_results['mean_similarity'] = 0.0
        test_results['std_similarity'] = 0.0
    
    if test_results['genre_match_accuracy']:
        test_results['mean_genre_match'] = np.mean(test_results['genre_match_accuracy'])
    else:
        test_results['mean_genre_match'] = 0.0
    
    if test_results['author_match_accuracy']:
        test_results['mean_author_match'] = np.mean(test_results['author_match_accuracy'])
    else:
        test_results['mean_author_match'] = 0.0
    
    return test_results

def get_similar_books(target_book: Dict[str, Any], 
                     books_with_embeddings: List[Dict[str, Any]], 
                     top_k: int = 10) -> List[Dict[str, Any]]:
    """Get similar books based on embedding similarity"""
    target_embedding = target_book.get('embedding', [])
    if not target_embedding:
        return []
    
    similarities = []
    
    for book in books_with_embeddings:
        if book.get('id') == target_book.get('id'):
            continue  # Skip the same book
        
        book_embedding = book.get('embedding', [])
        if not book_embedding:
            continue
        
        similarity = cosine_similarity(target_embedding, book_embedding)
        similarities.append((book, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k similar books
    return [{'book': book, 'similarity': sim} for book, sim in similarities[:top_k]]

def analyze_recommendation_quality(original_book: Dict[str, Any], 
                                 found_book: Dict[str, Any], 
                                 recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the quality of recommendations"""
    
    # Get genres from the original Goodreads book
    original_genres = set()
    if 'genres' in original_book and original_book['genres']:
        original_genres.update([g.lower() for g in original_book['genres']])
    
    found_genres = set(found_book.get('normalized_genres', []))
    original_author = original_book.get('author', '').lower()
    
    analysis = {
        'original_title': original_book.get('title', ''),
        'original_author': original_book.get('author', ''),
        'original_genres': list(original_genres),
        'found_title': found_book.get('title', ''),
        'found_author': found_book.get('primary_author', ''),
        'found_genres': list(found_genres),
        'recommendation_count': len(recommendations),
        'similarities': [],
        'genre_matches': 0,
        'author_matches': 0,
        'high_quality_recs': 0,  # Similarity > 0.8
        'medium_quality_recs': 0,  # Similarity 0.6-0.8
        'recommendations': []
    }
    
    for rec in recommendations:
        book = rec['book']
        similarity = rec['similarity']
        
        analysis['similarities'].append(similarity)
        
        # Check genre overlap
        rec_genres = set(book.get('normalized_genres', []))
        genre_overlap = len(original_genres.intersection(rec_genres)) > 0 or len(found_genres.intersection(rec_genres)) > 0
        if genre_overlap:
            analysis['genre_matches'] += 1
        
        # Check author match
        rec_author = book.get('primary_author', '').lower()
        if original_author in rec_author or rec_author in original_author:
            analysis['author_matches'] += 1
        
        # Quality categorization
        if similarity > 0.8:
            analysis['high_quality_recs'] += 1
        elif similarity > 0.6:
            analysis['medium_quality_recs'] += 1
        
        # Store recommendation details
        analysis['recommendations'].append({
            'title': book.get('title', ''),
            'author': book.get('primary_author', ''),
            'genres': book.get('normalized_genres', []),
            'similarity': similarity,
            'rating': book.get('average_rating', 0),
            'genre_match': genre_overlap
        })
    
    # Calculate rates
    total_recs = len(recommendations)
    analysis['genre_match_rate'] = analysis['genre_matches'] / total_recs if total_recs > 0 else 0
    analysis['author_match_rate'] = analysis['author_matches'] / total_recs if total_recs > 0 else 0
    analysis['high_quality_rate'] = analysis['high_quality_recs'] / total_recs if total_recs > 0 else 0
    
    return analysis

def test_genre_clustering(books_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Test how well embeddings cluster books by genre"""
    logger.info("Testing genre clustering quality...")
    
    # Group books by genre
    genre_books = defaultdict(list)
    for book in books_with_embeddings:
        if book.get('embedding'):
            for genre in book.get('normalized_genres', []):
                genre_books[genre].append(book)
    
    # Test clustering for genres with enough books
    clustering_results = {}
    
    for genre, books in genre_books.items():
        if len(books) < 5:  # Need at least 5 books for meaningful test
            continue
        
        # Calculate intra-genre similarities
        intra_similarities = []
        for i, book1 in enumerate(books[:20]):  # Limit to 20 books per genre
            for book2 in books[i+1:21]:
                sim = cosine_similarity(book1['embedding'], book2['embedding'])
                intra_similarities.append(sim)
        
        if intra_similarities:
            clustering_results[genre] = {
                'book_count': len(books),
                'mean_intra_similarity': np.mean(intra_similarities),
                'std_intra_similarity': np.std(intra_similarities),
                'max_intra_similarity': np.max(intra_similarities),
                'min_intra_similarity': np.min(intra_similarities)
            }
    
    return clustering_results

def generate_test_report(test_results: Dict[str, Any], 
                        clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive test report"""
    
    report = {
        'test_timestamp': datetime.now().isoformat(),
        'embedding_quality_assessment': {
            'overall_score': 'Unknown',
            'recommendation_accuracy': test_results.get('mean_similarity', 0),
            'genre_matching_accuracy': test_results.get('mean_genre_match', 0),
            'author_matching_accuracy': test_results.get('mean_author_match', 0),
            'books_tested': test_results.get('books_found_in_embeddings', 0),
            'total_recent_books': test_results.get('total_recent_books', 0)
        },
        'detailed_results': test_results,
        'genre_clustering': clustering_results,
        'recommendations': {
            'ready_for_production': False,
            'confidence_level': 'Low',
            'issues_found': [],
            'strengths': []
        }
    }
    
    # Assess overall quality
    mean_sim = test_results.get('mean_similarity', 0)
    genre_match = test_results.get('mean_genre_match', 0)
    books_tested = test_results.get('books_found_in_embeddings', 0)
    
    # Determine readiness
    issues = []
    strengths = []
    
    if mean_sim < 0.3:
        issues.append("Low average similarity scores indicate poor embedding quality")
    elif mean_sim > 0.6:
        strengths.append("Good average similarity scores")
    
    if genre_match < 0.3:
        issues.append("Poor genre matching in recommendations")
    elif genre_match > 0.5:
        strengths.append("Good genre matching accuracy")
    
    if books_tested < 3:
        issues.append("Insufficient test data - need more recent user reads")
    else:
        strengths.append(f"Tested with {books_tested} recent user reads")
    
    # Overall assessment
    if mean_sim > 0.5 and genre_match > 0.4 and books_tested >= 3:
        report['recommendations']['ready_for_production'] = True
        report['recommendations']['confidence_level'] = 'High'
        report['embedding_quality_assessment']['overall_score'] = 'Good'
    elif mean_sim > 0.3 and genre_match > 0.3:
        report['recommendations']['confidence_level'] = 'Medium'
        report['embedding_quality_assessment']['overall_score'] = 'Fair'
    else:
        report['embedding_quality_assessment']['overall_score'] = 'Poor'
    
    report['recommendations']['issues_found'] = issues
    report['recommendations']['strengths'] = strengths
    
    return report

def main():
    """Main testing process"""
    logger.info("=== Embedding Score Testing Process ===")
    
    # Load data
    books_with_embeddings = load_books_with_embeddings()
    goodreads_books = load_original_goodreads_data()
    
    if not books_with_embeddings:
        logger.error("No books with embeddings found")
        return
    
    if not goodreads_books:
        logger.error("No Goodreads data found")
        return
    
    # Get recent user books
    recent_books = get_recent_user_books(goodreads_books, months_back=18)
    
    if not recent_books:
        logger.warning("No recent books found. Trying with longer time period...")
        recent_books = get_recent_user_books(goodreads_books, months_back=36)
    
    if not recent_books:
        logger.error("No recent user reads found for testing")
        return
    
    # Test content-based recommendations
    logger.info("\n=== Testing Content-Based Recommendations ===")
    test_results = test_content_based_recommendations(recent_books, books_with_embeddings)
    
    # Test genre clustering
    logger.info("\n=== Testing Genre Clustering ===")
    clustering_results = test_genre_clustering(books_with_embeddings)
    
    # Generate comprehensive report
    logger.info("\n=== Generating Test Report ===")
    report = generate_test_report(test_results, clustering_results)
    
    # Save report
    with open('embedding_test_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Display results
    logger.info("\n=== Test Results Summary ===")
    logger.info(f"Overall Score: {report['embedding_quality_assessment']['overall_score']}")
    logger.info(f"Recommendation Accuracy: {report['embedding_quality_assessment']['recommendation_accuracy']:.3f}")
    logger.info(f"Genre Matching Accuracy: {report['embedding_quality_assessment']['genre_matching_accuracy']:.3f}")
    logger.info(f"Books Tested: {report['embedding_quality_assessment']['books_tested']}")
    logger.info(f"Ready for Production: {report['recommendations']['ready_for_production']}")
    logger.info(f"Confidence Level: {report['recommendations']['confidence_level']}")
    
    if report['recommendations']['strengths']:
        logger.info("\nStrengths:")
        for strength in report['recommendations']['strengths']:
            logger.info(f"  ✅ {strength}")
    
    if report['recommendations']['issues_found']:
        logger.info("\nIssues Found:")
        for issue in report['recommendations']['issues_found']:
            logger.info(f"  ⚠️  {issue}")
    
    # Show top clustering genres
    if clustering_results:
        logger.info("\nTop Genre Clustering Results:")
        sorted_genres = sorted(clustering_results.items(), 
                             key=lambda x: x[1]['mean_intra_similarity'], 
                             reverse=True)
        for genre, stats in sorted_genres[:5]:
            logger.info(f"  {genre}: {stats['mean_intra_similarity']:.3f} similarity "
                       f"({stats['book_count']} books)")
    
    logger.info("\nFiles created:")
    logger.info("  - embedding_test_report.json")
    
    return report

if __name__ == "__main__":
    main()