#!/usr/bin/env python3
"""
Collect 200 Horror books using Google Books API
"""

import requests
import json
import time
import random

def search_books(query, max_results=40):
    """Search for books using Google Books API"""
    base_url = "https://www.googleapis.com/books/v1/volumes"
    
    params = {
        'q': query,
        'maxResults': max_results,
        'orderBy': 'relevance',
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
    if published_date:
        try:
            year = int(published_date.split('-')[0])
        except:
            pass
    
    # Filter for books from 2000 onwards (to get modern horror)
    if year and year < 2000:
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
        'genres': ['Horror', 'Thriller']
    }
    
    # Add additional genres based on categories
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

def collect_horror_books():
    """Collect 200 Horror books"""
    print("Starting Horror book collection...")
    print("Collecting 200 Horror books...")
    
    horror_queries = [
        "horror fiction bestseller",
        "horror novels bestseller",
        "supernatural horror bestseller",
        "psychological horror bestseller",
        "gothic horror bestseller",
        "paranormal horror bestseller",
        "ghost stories bestseller",
        "vampire fiction bestseller",
        "zombie fiction bestseller",
        "werewolf fiction bestseller",
        "demon fiction bestseller",
        "haunted house fiction bestseller",
        "occult fiction bestseller",
        "dark fantasy bestseller",
        "body horror fiction",
        "cosmic horror bestseller",
        "folk horror fiction",
        "urban horror fiction",
        "monster fiction bestseller",
        "apocalyptic horror fiction",
        "Stephen King horror novels",
        "Clive Barker horror fiction",
        "Anne Rice horror novels",
        "Dean Koontz horror fiction",
        "Paul Tremblay horror novels",
        "Grady Hendrix horror fiction",
        "Shirley Jackson horror novels",
        "H.P. Lovecraft horror fiction",
        "Joe Hill horror novels",
        "Chuck Palahniuk horror fiction",
        "Thomas Ligotti horror fiction",
        "Ramsey Campbell horror novels",
        "Brian Evenson horror fiction",
        "Caitlin R Kiernan horror novels",
        "Laird Barron horror fiction"
    ]
    
    collected_books = []
    seen_titles = set()
    
    for query in horror_queries:
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
                print(f"  Collected {len(collected_books)} horror books...")
    
    print(f"Collected {len(collected_books)} Horror books")
    
    # Save to file
    with open('collected_horror_books.json', 'w', encoding='utf-8') as f:
        json.dump(collected_books, f, indent=2, ensure_ascii=False)
    
    return collected_books

if __name__ == "__main__":
    horror_books = collect_horror_books()
    print(f"\nHorror collection complete!")
    print(f"Total Horror books collected: {len(horror_books)}")
    print("Saved to: collected_horror_books.json")