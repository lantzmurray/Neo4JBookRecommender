#!/usr/bin/env python3
"""
Collect 200 New Release books from the last 6 months using Google Books API
"""

import requests
import json
import time
import random
from datetime import datetime, timedelta

def search_books(query, max_results=40):
    """Search for books using Google Books API"""
    base_url = "https://www.googleapis.com/books/v1/volumes"
    
    params = {
        'q': query,
        'maxResults': max_results,
        'orderBy': 'newest',
        'printType': 'books',
        'langRestrict': 'en'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error searching for '{query}': {e}")
        return None

def extract_book_info(item):
    """Extract relevant book information from API response"""
    volume_info = item.get('volumeInfo', {})
    
    # Get publication year
    published_date = volume_info.get('publishedDate', '')
    year = None
    month = None
    if published_date:
        try:
            date_parts = published_date.split('-')
            year = int(date_parts[0])
            if len(date_parts) > 1:
                month = int(date_parts[1])
        except:
            pass
    
    # Filter for recent books (last 6 months from 2025)
    cutoff_date = datetime(2024, 7, 1)  # July 2024 onwards
    if year and year >= 2024:
        if year == 2024 and month and month < 7:
            return None
    elif not year or year < 2024:
        return None
    
    # Extract basic info
    book_info = {
        'title': volume_info.get('title', ''),
        'author': ', '.join(volume_info.get('authors', [])),
        'isbn': '',
        'isbn13': '',
        'year_published': year,
        'publisher': volume_info.get('publisher', ''),
        'pages': volume_info.get('pageCount', 0),
        'average_rating': volume_info.get('averageRating', 0),
        'ratings_count': volume_info.get('ratingsCount', 0),
        'description': volume_info.get('description', ''),
        'categories': volume_info.get('categories', []),
        'google_id': item.get('id', ''),
        'source': 'google_books',
        'genres': ['New Release']
    }
    
    # Add genre based on categories
    categories = volume_info.get('categories', [])
    if categories:
        book_info['genres'].extend(categories[:2])  # Add up to 2 categories
    
    # Extract ISBNs
    identifiers = volume_info.get('industryIdentifiers', [])
    for identifier in identifiers:
        if identifier.get('type') == 'ISBN_10':
            book_info['isbn'] = identifier.get('identifier', '')
        elif identifier.get('type') == 'ISBN_13':
            book_info['isbn13'] = identifier.get('identifier', '')
    
    # Skip if no title or author
    if not book_info['title'] or not book_info['author']:
        return None
    
    return book_info

def collect_new_releases():
    """Collect 200 New Release books from the last 6 months"""
    print("Starting New Release book collection...")
    print("Collecting 200 New Release books from the last 6 months...")
    
    new_release_queries = [
        "new releases 2024 bestseller fiction",
        "new books 2024 popular fiction",
        "recent releases bestseller 2024",
        "new novels 2024 bestseller list",
        "latest books 2024 fiction bestseller",
        "new releases mystery 2024",
        "new books romance 2024 bestseller",
        "recent fiction 2024 popular",
        "new releases thriller 2024",
        "latest novels 2024 bestseller",
        "new books science fiction 2024",
        "recent releases horror 2024",
        "new novels fantasy 2024 bestseller",
        "latest books literary fiction 2024",
        "new releases contemporary fiction 2024",
        "recent books bestseller 2024",
        "new publications 2024 fiction",
        "latest releases popular 2024",
        "new books award winning 2024",
        "recent novels bestseller 2024"
    ]
    
    collected_books = []
    seen_titles = set()
    
    for query in new_release_queries:
        if len(collected_books) >= 200:
            break
            
        print(f"Searching: {query}")
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(1, 3))
        
        data = search_books(query)
        if not data or 'items' not in data:
            continue
        
        for item in data['items']:
            if len(collected_books) >= 200:
                break
                
            book_info = extract_book_info(item)
            if not book_info:
                continue
            
            # Check for duplicates
            title_key = f"{book_info['title'].lower()}_{book_info['author'].lower()}"
            if title_key in seen_titles:
                continue
            
            seen_titles.add(title_key)
            collected_books.append(book_info)
            
            # Progress update every 25 books
            if len(collected_books) % 25 == 0:
                print(f"  Collected {len(collected_books)} new release books...")
    
    print(f"Collected {len(collected_books)} New Release books")
    
    # Save to file
    with open('collected_new_releases.json', 'w', encoding='utf-8') as f:
        json.dump(collected_books, f, indent=2, ensure_ascii=False)
    
    return collected_books

if __name__ == "__main__":
    new_release_books = collect_new_releases()
    print(f"\nNew Release collection complete!")
    print(f"Total New Release books collected: {len(new_release_books)}")
    print("Saved to: collected_new_releases.json")