#!/usr/bin/env python3
"""
Merge all book collections with Goodreads data and prepare final dataset
"""

import json
import pandas as pd
from datetime import datetime
import re

def load_json_file(filename):
    """Load JSON file safely"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return []
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

def normalize_book_data(book, source):
    """Normalize book data structure"""
    normalized = {
        'title': str(book.get('title', '')).strip(),
        'author': str(book.get('author', '')).strip(),
        'isbn': str(book.get('isbn', '')).strip(),
        'isbn13': str(book.get('isbn13', '')).strip(),
        'year_published': book.get('year_published', 0),
        'publisher': str(book.get('publisher', '')).strip(),
        'pages': book.get('pages', 0),
        'average_rating': float(book.get('average_rating', 0)),
        'ratings_count': int(book.get('ratings_count', 0)),
        'description': str(book.get('description', '')).strip(),
        'categories': book.get('categories', []),
        'genres': book.get('genres', []),
        'source': source,
        'google_id': book.get('google_id', ''),
        'goodreads_id': book.get('goodreads_id', ''),
        'user_rating': book.get('user_rating', 0),
        'user_read_date': book.get('user_read_date', ''),
        'user_shelves': book.get('user_shelves', [])
    }
    
    # Ensure year is integer
    try:
        normalized['year_published'] = int(normalized['year_published'])
    except:
        normalized['year_published'] = 0
    
    # Ensure pages is integer
    try:
        normalized['pages'] = int(normalized['pages'])
    except:
        normalized['pages'] = 0
    
    return normalized

def calculate_popularity_score(book):
    """Calculate popularity score for ranking"""
    rating = float(book.get('average_rating', 0))
    count = int(book.get('ratings_count', 0))
    
    # Weighted score: rating * log(count + 1)
    import math
    score = rating * math.log(count + 1) if count > 0 else rating
    
    # Boost for user favorites
    if book.get('user_rating', 0) >= 4:
        score *= 1.2
    
    return round(score, 3)

def create_deduplication_key(book):
    """Create key for deduplication"""
    title = re.sub(r'[^\w\s]', '', book['title'].lower())
    author = re.sub(r'[^\w\s]', '', book['author'].lower())
    return f"{title}_{author}"

def merge_all_collections():
    """Merge all book collections into a single dataset"""
    print("Starting merge process...")
    
    # Load all collections
    collections = {
        'goodreads': load_json_file('goodreads_books.json'),
        'mystery': load_json_file('collected_mystery_books.json'),
        'mystery_additional': load_json_file('collected_mystery_books_additional.json'),
        'fantasy': load_json_file('collected_fantasy_books.json'),
        'scifi': load_json_file('collected_scifi_books.json'),
        'romance': load_json_file('collected_romance_books.json'),
        'thriller': load_json_file('collected_thriller_books.json'),
        'horror': load_json_file('collected_horror_books.json'),
        'historical': load_json_file('collected_historical_books.json'),
        'biography': load_json_file('collected_biography_books.json'),
        'nonfiction': load_json_file('collected_nonfiction_books.json'),
        'splatterpunk': load_json_file('collected_splatterpunk_books.json'),
        'new_releases': load_json_file('collected_new_releases.json'),
        'crime': load_json_file('collected_crime_books.json'),
        'suspense': load_json_file('collected_suspense_books.json')
    }
    
    print("Collection sizes:")
    for name, books in collections.items():
        print(f"  {name.capitalize()}: {len(books)} books")
    
    # Normalize all books
    all_books = []
    for source, books in collections.items():
        for book in books:
            normalized = normalize_book_data(book, source)
            if normalized['title'] and normalized['author']:
                all_books.append(normalized)
    
    print(f"\nTotal books before deduplication: {len(all_books)}")
    
    # Deduplicate books
    seen_keys = {}
    unique_books = []
    
    for book in all_books:
        key = create_deduplication_key(book)
        
        if key in seen_keys:
            # Merge data from duplicate
            existing = seen_keys[key]
            
            # Keep the one with more data or higher rating
            if (book['ratings_count'] > existing['ratings_count'] or 
                book['average_rating'] > existing['average_rating'] or
                len(book['description']) > len(existing['description'])):
                
                # Merge genres and categories
                existing['genres'] = list(set(existing['genres'] + book['genres']))
                existing['categories'] = list(set(existing['categories'] + book['categories']))
                
                # Update if this version has more data
                if book['ratings_count'] > existing['ratings_count']:
                    existing.update({
                        'average_rating': book['average_rating'],
                        'ratings_count': book['ratings_count'],
                        'description': book['description'] if len(book['description']) > len(existing['description']) else existing['description'],
                        'publisher': book['publisher'] if book['publisher'] else existing['publisher'],
                        'pages': book['pages'] if book['pages'] > existing['pages'] else existing['pages']
                    })
        else:
            seen_keys[key] = book
            unique_books.append(book)
    
    print(f"Unique books after deduplication: {len(unique_books)}")
    
    # Calculate popularity scores
    for book in unique_books:
        book['popularity_score'] = calculate_popularity_score(book)
    
    # Sort by popularity score
    unique_books.sort(key=lambda x: x['popularity_score'], reverse=True)
    
    # Genre distribution
    genre_counts = {}
    for book in unique_books:
        for genre in book['genres']:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Create summary
    summary = {
        'total_books': len(unique_books),
        'genre_distribution': genre_counts,
        'source_distribution': {},
        'year_range': {
            'min': min([b['year_published'] for b in unique_books if b['year_published'] > 0], default=0),
            'max': max([b['year_published'] for b in unique_books if b['year_published'] > 0], default=0)
        },
        'rating_stats': {
            'avg_rating': round(sum([b['average_rating'] for b in unique_books if b['average_rating'] > 0]) / 
                              len([b for b in unique_books if b['average_rating'] > 0]), 2),
            'avg_ratings_count': round(sum([b['ratings_count'] for b in unique_books]) / len(unique_books))
        },
        'merge_date': datetime.now().isoformat()
    }
    
    # Source distribution
    for book in unique_books:
        source = book['source']
        summary['source_distribution'][source] = summary['source_distribution'].get(source, 0) + 1
    
    # Save merged dataset
    with open('merged_book_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(unique_books, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    df = pd.DataFrame(unique_books)
    df.to_csv('merged_book_dataset.csv', index=False, encoding='utf-8')
    
    # Save summary
    with open('merge_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nMerge complete!")
    print(f"Final dataset: {summary['total_books']} unique books")
    print(f"Genre distribution:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count}")
    print(f"Year range: {summary['year_range']['min']} - {summary['year_range']['max']}")
    print(f"Average rating: {summary['rating_stats']['avg_rating']}")
    
    return unique_books, summary

if __name__ == "__main__":
    books, summary = merge_all_collections()
    print(f"\nFiles saved:")
    print(f"  - merged_book_dataset.json ({len(books)} books)")
    print(f"  - merged_book_dataset.csv")
    print(f"  - merge_summary.json")