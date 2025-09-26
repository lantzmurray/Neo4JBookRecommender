#!/usr/bin/env python3
"""
Book Data Enrichment Script for Recommendation System
Enhances merged book data with additional metadata and features
"""

import json
import pandas as pd
import re
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

def load_merged_data():
    """Load the merged book dataset"""
    try:
        with open('merged_book_dataset.json', 'r', encoding='utf-8') as f:
            books = json.load(f)
        print(f"Loaded {len(books)} books from merged dataset")
        return books
    except FileNotFoundError:
        print("Error: merged_book_dataset.json not found")
        return []

def normalize_genres(books):
    """Normalize and standardize genre information"""
    print("Normalizing genres...")
    
    # Genre mapping for standardization
    genre_mapping = {
        'fiction': ['fiction', 'literary fiction', 'contemporary fiction', 'general fiction'],
        'romance': ['romance', 'contemporary romance', 'historical romance', 'paranormal romance'],
        'mystery': ['mystery', 'mystery thriller', 'cozy mystery', 'detective', 'crime'],
        'thriller': ['thriller', 'suspense', 'psychological thriller', 'action thriller'],
        'horror': ['horror', 'supernatural horror', 'gothic horror', 'paranormal horror'],
        'fantasy': ['fantasy', 'epic fantasy', 'urban fantasy', 'high fantasy', 'dark fantasy'],
        'science_fiction': ['science fiction', 'sci-fi', 'dystopian', 'space opera', 'cyberpunk'],
        'historical': ['historical fiction', 'historical', 'period fiction'],
        'young_adult': ['young adult', 'ya', 'teen fiction'],
        'literary': ['literary fiction', 'literary', 'classics', 'literature'],
        'adventure': ['adventure', 'action adventure', 'survival'],
        'biography': ['biography', 'memoir', 'autobiography'],
        'non_fiction': ['non-fiction', 'nonfiction', 'self-help', 'history'],
        'children': ['children', 'juvenile', 'picture book'],
        'comedy': ['humor', 'comedy', 'humorous fiction'],
        'drama': ['drama', 'family drama', 'domestic fiction']
    }
    
    # Create reverse mapping
    reverse_mapping = {}
    for standard_genre, variants in genre_mapping.items():
        for variant in variants:
            reverse_mapping[variant.lower()] = standard_genre
    
    for book in books:
        if 'genres' in book and book['genres']:
            normalized_genres = set()
            for genre in book['genres']:
                if isinstance(genre, str):
                    genre_lower = genre.lower().strip()
                    # Direct mapping
                    if genre_lower in reverse_mapping:
                        normalized_genres.add(reverse_mapping[genre_lower])
                    # Partial matching
                    else:
                        for variant, standard in reverse_mapping.items():
                            if variant in genre_lower or genre_lower in variant:
                                normalized_genres.add(standard)
                                break
                        else:
                            # Keep original if no mapping found
                            normalized_genres.add(genre_lower.replace(' ', '_'))
            
            book['normalized_genres'] = list(normalized_genres)
        else:
            book['normalized_genres'] = []
    
    return books

def extract_keywords_from_description(description):
    """Extract keywords from book description"""
    if not description or not isinstance(description, str):
        return []
    
    # Common words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Extract words (remove punctuation, convert to lowercase)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', description.lower())
    keywords = [word for word in words if word not in stop_words]
    
    # Return most common keywords
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(10)]

def calculate_book_features(books):
    """Calculate additional features for recommendation system"""
    print("Calculating book features...")
    
    current_year = datetime.now().year
    
    for book in books:
        # Age of book
        year_published = book.get('year_published', current_year)
        if isinstance(year_published, str):
            try:
                year_published = int(year_published)
            except ValueError:
                year_published = current_year
        
        book['book_age'] = current_year - year_published
        
        # Recency score (newer books get higher scores)
        book['recency_score'] = max(0, 10 - (book['book_age'] / 2))
        
        # Rating reliability (based on number of ratings)
        ratings_count = book.get('ratings_count', 0)
        if isinstance(ratings_count, str):
            try:
                ratings_count = int(ratings_count.replace(',', ''))
            except ValueError:
                ratings_count = 0
        
        book['ratings_count'] = ratings_count
        book['rating_reliability'] = min(10, ratings_count / 100)  # Scale 0-10
        
        # Popularity tier
        if ratings_count >= 10000:
            book['popularity_tier'] = 'high'
        elif ratings_count >= 1000:
            book['popularity_tier'] = 'medium'
        else:
            book['popularity_tier'] = 'low'
        
        # Extract keywords from description
        book['description_keywords'] = extract_keywords_from_description(
            book.get('description', '')
        )
        
        # Page count category
        pages = book.get('pages', 0)
        if isinstance(pages, str):
            try:
                pages = int(pages)
            except ValueError:
                pages = 0
        
        book['pages'] = pages
        if pages == 0:
            book['length_category'] = 'unknown'
        elif pages < 200:
            book['length_category'] = 'short'
        elif pages < 400:
            book['length_category'] = 'medium'
        else:
            book['length_category'] = 'long'
        
        # Clean and standardize author names
        author = book.get('author', '')
        if isinstance(author, str) and author:
            # Clean author name
            clean_author = re.sub(r'\s+', ' ', author.strip())
            clean_author = re.sub(r'\s*\(.*?\)\s*', '', clean_author)  # Remove parentheses
            book['authors'] = [clean_author] if clean_author else ['Unknown']
        else:
            book['authors'] = ['Unknown']
        
        book['primary_author'] = book['authors'][0]
        
        # Create a composite score for ranking
        avg_rating = book.get('average_rating', 0)
        if isinstance(avg_rating, str):
            try:
                avg_rating = float(avg_rating)
            except ValueError:
                avg_rating = 0
        
        book['average_rating'] = avg_rating
        
        # Composite recommendation score
        book['recommendation_score'] = (
            avg_rating * 0.4 +  # Rating weight
            book['rating_reliability'] * 0.2 +  # Reliability weight
            book['recency_score'] * 0.2 +  # Recency weight
            (book.get('popularity_score', 0) / 10) * 0.2  # Popularity weight
        )
    
    return books

def create_author_profiles(books):
    """Create author profiles with statistics"""
    print("Creating author profiles...")
    
    author_stats = defaultdict(lambda: {
        'books': [],
        'total_books': 0,
        'avg_rating': 0,
        'total_ratings': 0,
        'genres': set(),
        'years_active': set()
    })
    
    for i, book in enumerate(books):
        # Create a unique book identifier if 'id' doesn't exist
        book_id = book.get('id', f"{book.get('title', 'unknown')}_{i}")
        book['id'] = book_id  # Add id to book for consistency
        
        for author in book.get('authors', []):
            if author and author != 'Unknown':
                stats = author_stats[author]
                stats['books'].append(book_id)
                stats['total_books'] += 1
                stats['total_ratings'] += book.get('ratings_count', 0)
                stats['genres'].update(book.get('normalized_genres', []))
                if book.get('year_published'):
                    stats['years_active'].add(book['year_published'])
    
    # Calculate averages
    for author, stats in author_stats.items():
        if stats['total_books'] > 0:
            # Calculate average rating across all books
            total_weighted_rating = sum(
                book.get('average_rating', 0) * book.get('ratings_count', 1)
                for book in books
                if book.get('primary_author') == author
            )
            total_weight = sum(
                book.get('ratings_count', 1)
                for book in books
                if book.get('primary_author') == author
            )
            stats['avg_rating'] = total_weighted_rating / total_weight if total_weight > 0 else 0
            stats['genres'] = list(stats['genres'])
            stats['years_active'] = list(stats['years_active'])
            stats['career_span'] = max(stats['years_active']) - min(stats['years_active']) if len(stats['years_active']) > 1 else 0
    
    return dict(author_stats)

def create_genre_profiles(books):
    """Create genre profiles with statistics"""
    print("Creating genre profiles...")
    
    genre_stats = defaultdict(lambda: {
        'books': [],
        'total_books': 0,
        'avg_rating': 0,
        'avg_pages': 0,
        'total_ratings': 0,
        'top_authors': Counter(),
        'year_distribution': Counter()
    })
    
    for book in books:
        # Ensure book has an ID (should be set by create_author_profiles)
        book_id = book.get('id', f"{book.get('title', 'unknown')}_{hash(str(book))}")
        book['id'] = book_id
        
        for genre in book.get('normalized_genres', []):
            stats = genre_stats[genre]
            stats['books'].append(book_id)
            stats['total_books'] += 1
            stats['total_ratings'] += book.get('ratings_count', 0)
            stats['top_authors'][book.get('primary_author', 'Unknown')] += 1
            stats['year_distribution'][book.get('year_published', 'Unknown')] += 1
    
    # Calculate averages
    for genre, stats in genre_stats.items():
        if stats['total_books'] > 0:
            genre_books = [book for book in books if genre in book.get('normalized_genres', [])]
            
            # Weighted average rating
            total_weighted_rating = sum(
                book.get('average_rating', 0) * book.get('ratings_count', 1)
                for book in genre_books
            )
            total_weight = sum(book.get('ratings_count', 1) for book in genre_books)
            stats['avg_rating'] = total_weighted_rating / total_weight if total_weight > 0 else 0
            
            # Average pages
            pages_list = [book.get('pages', 0) for book in genre_books if book.get('pages', 0) > 0]
            stats['avg_pages'] = sum(pages_list) / len(pages_list) if pages_list else 0
            
            # Top authors (top 5)
            stats['top_authors'] = dict(stats['top_authors'].most_common(5))
            
            # Recent years (last 10 years)
            recent_years = {year: count for year, count in stats['year_distribution'].items() 
                          if isinstance(year, int) and year >= datetime.now().year - 10}
            stats['recent_activity'] = sum(recent_years.values())
    
    return dict(genre_stats)

def save_enriched_data(books, author_profiles, genre_profiles):
    """Save enriched data to files"""
    print("Saving enriched data...")
    
    # Save enriched books
    with open('enriched_book_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(books, f, indent=2, ensure_ascii=False)
    
    # Save as CSV for analysis
    df = pd.DataFrame(books)
    df.to_csv('enriched_book_dataset.csv', index=False, encoding='utf-8')
    
    # Save author profiles
    with open('author_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(author_profiles, f, indent=2, ensure_ascii=False)
    
    # Save genre profiles
    with open('genre_profiles.json', 'w', encoding='utf-8') as f:
        json.dump(genre_profiles, f, indent=2, ensure_ascii=False)
    
    # Create enrichment summary
    summary = {
        'total_books': len(books),
        'total_authors': len(author_profiles),
        'total_genres': len(genre_profiles),
        'enrichment_features': [
            'normalized_genres',
            'description_keywords',
            'book_age',
            'recency_score',
            'rating_reliability',
            'popularity_tier',
            'length_category',
            'recommendation_score'
        ],
        'genre_distribution': {genre: stats['total_books'] for genre, stats in genre_profiles.items()},
        'top_authors_by_book_count': {
            author: stats['total_books'] 
            for author, stats in sorted(author_profiles.items(), 
                                      key=lambda x: x[1]['total_books'], 
                                      reverse=True)[:10]
        },
        'rating_distribution': {
            'avg_rating_overall': np.mean([book.get('average_rating', 0) for book in books]),
            'books_with_high_ratings': len([book for book in books if book.get('average_rating', 0) >= 4.0]),
            'books_with_many_ratings': len([book for book in books if book.get('ratings_count', 0) >= 1000])
        },
        'enrichment_timestamp': datetime.now().isoformat()
    }
    
    with open('enrichment_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

def main():
    """Main enrichment process"""
    print("=== Book Data Enrichment Process ===\n")
    
    # Load merged data
    books = load_merged_data()
    if not books:
        return
    
    # Normalize genres
    books = normalize_genres(books)
    
    # Calculate additional features
    books = calculate_book_features(books)
    
    # Create profiles
    author_profiles = create_author_profiles(books)
    genre_profiles = create_genre_profiles(books)
    
    # Save enriched data
    summary = save_enriched_data(books, author_profiles, genre_profiles)
    
    print("\n=== Enrichment Complete ===")
    print(f"Total books enriched: {summary['total_books']}")
    print(f"Total authors profiled: {summary['total_authors']}")
    print(f"Total genres identified: {summary['total_genres']}")
    print(f"Average rating across all books: {summary['rating_distribution']['avg_rating_overall']:.2f}")
    print(f"Books with 4+ rating: {summary['rating_distribution']['books_with_high_ratings']}")
    print(f"Books with 1000+ ratings: {summary['rating_distribution']['books_with_many_ratings']}")
    
    print("\nTop 5 genres by book count:")
    for genre, count in list(summary['genre_distribution'].items())[:5]:
        print(f"  {genre}: {count} books")
    
    print("\nFiles created:")
    print("  - enriched_book_dataset.json")
    print("  - enriched_book_dataset.csv")
    print("  - author_profiles.json")
    print("  - genre_profiles.json")
    print("  - enrichment_summary.json")

if __name__ == "__main__":
    main()