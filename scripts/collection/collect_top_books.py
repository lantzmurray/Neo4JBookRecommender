import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
import os

class BookCollector:
    def __init__(self):
        self.collected_books = defaultdict(list)
        self.all_books = []
        
    def search_google_books(self, query, max_results=40, start_index=0):
        """Search Google Books API for books matching the query."""
        base_url = "https://www.googleapis.com/books/v1/volumes"
        
        params = {
            'q': query,
            'maxResults': min(max_results, 40),  # API limit is 40
            'startIndex': start_index,
            'orderBy': 'relevance',
            'printType': 'books',
            'langRestrict': 'en'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching Google Books: {e}")
            return None
    
    def extract_book_info(self, item, genre):
        """Extract relevant book information from Google Books API response."""
        try:
            volume_info = item.get('volumeInfo', {})
            
            # Extract basic info
            title = volume_info.get('title', '')
            authors = volume_info.get('authors', [])
            author = ', '.join(authors) if authors else 'Unknown'
            
            # Extract identifiers
            isbn = ''
            isbn13 = ''
            industry_identifiers = volume_info.get('industryIdentifiers', [])
            for identifier in industry_identifiers:
                if identifier.get('type') == 'ISBN_10':
                    isbn = identifier.get('identifier', '')
                elif identifier.get('type') == 'ISBN_13':
                    isbn13 = identifier.get('identifier', '')
            
            # Extract publication info
            published_date = volume_info.get('publishedDate', '')
            year_published = None
            if published_date:
                try:
                    if len(published_date) >= 4:
                        year_published = int(published_date[:4])
                except:
                    pass
            
            # Extract other details
            page_count = volume_info.get('pageCount', 0)
            average_rating = volume_info.get('averageRating', 0)
            ratings_count = volume_info.get('ratingsCount', 0)
            publisher = volume_info.get('publisher', '')
            description = volume_info.get('description', '')
            
            # Extract categories/genres
            categories = volume_info.get('categories', [])
            
            book_data = {
                'title': title,
                'author': author,
                'isbn': isbn,
                'isbn13': isbn13,
                'year_published': year_published,
                'publisher': publisher,
                'pages': page_count,
                'average_rating': average_rating,
                'ratings_count': ratings_count,
                'description': description,
                'categories': categories,
                'genres': [genre],
                'source': 'google_books',
                'google_id': item.get('id', '')
            }
            
            return book_data
            
        except Exception as e:
            print(f"Error extracting book info: {e}")
            return None
    
    def collect_horror_books(self, target_count=500):
        """Collect horror books from multiple search queries."""
        print(f"Collecting {target_count} Horror books...")
        
        horror_queries = [
            'horror fiction bestseller 2019..2025',
            'horror novels Stephen King bestseller',
            'psychological horror thriller bestseller',
            'supernatural horror fiction popular',
            'ghost stories horror bestseller',
            'vampire horror fiction popular',
            'zombie apocalypse horror bestseller',
            'haunted house horror fiction',
            'demon possession horror novels',
            'werewolf horror fiction bestseller',
            'cosmic horror Lovecraft inspired',
            'body horror fiction bestseller',
            'slasher horror novels popular',
            'occult horror fiction bestseller',
            'monster horror creature fiction'
        ]
        
        collected = 0
        for query in horror_queries:
            if collected >= target_count:
                break
                
            print(f"Searching: {query}")
            
            # Search multiple pages for each query
            for start_index in range(0, 200, 40):  # 5 pages max per query
                if collected >= target_count:
                    break
                    
                result = self.search_google_books(query, max_results=40, start_index=start_index)
                if not result or 'items' not in result:
                    break
                
                for item in result['items']:
                    if collected >= target_count:
                        break
                        
                    book_info = self.extract_book_info(item, 'Horror')
                    if book_info and book_info['year_published'] and book_info['year_published'] >= 2019:
                        # Check for duplicates
                        if not self.is_duplicate(book_info):
                            self.collected_books['Horror'].append(book_info)
                            self.all_books.append(book_info)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Collected {collected} horror books...")
                
                time.sleep(0.5)  # Rate limiting
        
        print(f"Collected {len(self.collected_books['Horror'])} Horror books")
        return self.collected_books['Horror']
    
    def collect_thriller_books(self, target_count=500):
        """Collect thriller books from multiple search queries."""
        print(f"Collecting {target_count} Thriller books...")
        
        thriller_queries = [
            'thriller fiction bestseller 2019..2025',
            'psychological thriller bestseller',
            'suspense thriller novels popular',
            'crime thriller fiction bestseller',
            'mystery thriller bestseller',
            'legal thriller fiction popular',
            'medical thriller bestseller',
            'techno thriller fiction bestseller',
            'political thriller novels popular',
            'domestic thriller fiction bestseller',
            'spy thriller espionage bestseller',
            'action thriller adventure fiction',
            'conspiracy thriller bestseller',
            'serial killer thriller fiction',
            'kidnapping thriller bestseller'
        ]
        
        collected = 0
        for query in thriller_queries:
            if collected >= target_count:
                break
                
            print(f"Searching: {query}")
            
            for start_index in range(0, 200, 40):
                if collected >= target_count:
                    break
                    
                result = self.search_google_books(query, max_results=40, start_index=start_index)
                if not result or 'items' not in result:
                    break
                
                for item in result['items']:
                    if collected >= target_count:
                        break
                        
                    book_info = self.extract_book_info(item, 'Thriller')
                    if book_info and book_info['year_published'] and book_info['year_published'] >= 2019:
                        if not self.is_duplicate(book_info):
                            self.collected_books['Thriller'].append(book_info)
                            self.all_books.append(book_info)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Collected {collected} thriller books...")
                
                time.sleep(0.5)
        
        print(f"Collected {len(self.collected_books['Thriller'])} Thriller books")
        return self.collected_books['Thriller']
    
    def collect_dystopian_books(self, target_count=500):
        """Collect dystopian books from multiple search queries."""
        print(f"Collecting {target_count} Dystopian books...")
        
        dystopian_queries = [
            'dystopian fiction bestseller 2019..2025',
            'post apocalyptic fiction bestseller',
            'dystopian society novels popular',
            'future dystopia fiction bestseller',
            'totalitarian dystopian fiction',
            'cyberpunk dystopian bestseller',
            'climate dystopian fiction popular',
            'pandemic dystopian novels bestseller',
            'surveillance dystopian fiction',
            'rebellion dystopian bestseller',
            'artificial intelligence dystopian fiction',
            'genetic dystopian novels bestseller',
            'corporate dystopian fiction popular',
            'environmental dystopian bestseller',
            'social dystopian fiction novels'
        ]
        
        collected = 0
        for query in dystopian_queries:
            if collected >= target_count:
                break
                
            print(f"Searching: {query}")
            
            for start_index in range(0, 200, 40):
                if collected >= target_count:
                    break
                    
                result = self.search_google_books(query, max_results=40, start_index=start_index)
                if not result or 'items' not in result:
                    break
                
                for item in result['items']:
                    if collected >= target_count:
                        break
                        
                    book_info = self.extract_book_info(item, 'Dystopian')
                    if book_info and book_info['year_published'] and book_info['year_published'] >= 2019:
                        if not self.is_duplicate(book_info):
                            self.collected_books['Dystopian'].append(book_info)
                            self.all_books.append(book_info)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Collected {collected} dystopian books...")
                
                time.sleep(0.5)
        
        print(f"Collected {len(self.collected_books['Dystopian'])} Dystopian books")
        return self.collected_books['Dystopian']
    
    def collect_fiction_books(self, target_count=500):
        """Collect general fiction books from multiple search queries."""
        print(f"Collecting {target_count} Fiction books...")
        
        fiction_queries = [
            'literary fiction bestseller 2019..2025',
            'contemporary fiction bestseller',
            'historical fiction novels popular',
            'science fiction bestseller recent',
            'fantasy fiction bestseller popular',
            'adventure fiction novels bestseller',
            'family saga fiction bestseller',
            'coming of age fiction popular',
            'women fiction bestseller recent',
            'book club fiction bestseller',
            'award winning fiction bestseller',
            'debut fiction novels bestseller',
            'international fiction bestseller',
            'upmarket fiction popular novels',
            'mainstream fiction bestseller'
        ]
        
        collected = 0
        for query in fiction_queries:
            if collected >= target_count:
                break
                
            print(f"Searching: {query}")
            
            for start_index in range(0, 200, 40):
                if collected >= target_count:
                    break
                    
                result = self.search_google_books(query, max_results=40, start_index=start_index)
                if not result or 'items' not in result:
                    break
                
                for item in result['items']:
                    if collected >= target_count:
                        break
                        
                    book_info = self.extract_book_info(item, 'Fiction')
                    if book_info and book_info['year_published'] and book_info['year_published'] >= 2019:
                        if not self.is_duplicate(book_info):
                            self.collected_books['Fiction'].append(book_info)
                            self.all_books.append(book_info)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Collected {collected} fiction books...")
                
                time.sleep(0.5)
        
        print(f"Collected {len(self.collected_books['Fiction'])} Fiction books")
        return self.collected_books['Fiction']
    
    def collect_romance_books(self, target_count=500):
        """Collect romance books from multiple search queries."""
        print(f"Collecting {target_count} Romance books...")
        
        romance_queries = [
            'romance fiction bestseller 2019..2025',
            'contemporary romance bestseller',
            'romantic comedy fiction popular',
            'historical romance novels bestseller',
            'paranormal romance fiction popular',
            'romantic suspense bestseller',
            'new adult romance fiction',
            'enemies to lovers romance bestseller',
            'second chance romance popular',
            'workplace romance fiction bestseller',
            'small town romance novels popular',
            'billionaire romance fiction bestseller',
            'sports romance novels popular',
            'fake dating romance bestseller',
            'friends to lovers romance fiction'
        ]
        
        collected = 0
        for query in romance_queries:
            if collected >= target_count:
                break
                
            print(f"Searching: {query}")
            
            for start_index in range(0, 200, 40):
                if collected >= target_count:
                    break
                    
                result = self.search_google_books(query, max_results=40, start_index=start_index)
                if not result or 'items' not in result:
                    break
                
                for item in result['items']:
                    if collected >= target_count:
                        break
                        
                    book_info = self.extract_book_info(item, 'Romance')
                    if book_info and book_info['year_published'] and book_info['year_published'] >= 2019:
                        if not self.is_duplicate(book_info):
                            self.collected_books['Romance'].append(book_info)
                            self.all_books.append(book_info)
                            collected += 1
                            
                            if collected % 50 == 0:
                                print(f"  Collected {collected} romance books...")
                
                time.sleep(0.5)
        
        print(f"Collected {len(self.collected_books['Romance'])} Romance books")
        return self.collected_books['Romance']
    
    def is_duplicate(self, new_book):
        """Check if a book is already in the collection."""
        for existing_book in self.all_books:
            # Check by title and author
            if (new_book['title'].lower() == existing_book['title'].lower() and 
                new_book['author'].lower() == existing_book['author'].lower()):
                return True
            
            # Check by ISBN if available
            if (new_book['isbn'] and existing_book['isbn'] and 
                new_book['isbn'] == existing_book['isbn']):
                return True
            
            if (new_book['isbn13'] and existing_book['isbn13'] and 
                new_book['isbn13'] == existing_book['isbn13']):
                return True
        
        return False
    
    def save_collections(self):
        """Save all collected books to files."""
        # Save by genre
        for genre, books in self.collected_books.items():
            filename = f"collected_{genre.lower()}_books.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(books, f, indent=2, ensure_ascii=False, default=str)
            print(f"Saved {len(books)} {genre} books to {filename}")
        
        # Save all books combined
        with open('all_collected_books.json', 'w', encoding='utf-8') as f:
            json.dump(self.all_books, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved {len(self.all_books)} total books to all_collected_books.json")
        
        # Create summary
        summary = {
            'total_books': len(self.all_books),
            'by_genre': {genre: len(books) for genre, books in self.collected_books.items()},
            'collection_date': datetime.now().isoformat()
        }
        
        with open('collection_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary

def main():
    collector = BookCollector()
    
    print("Starting comprehensive book collection...")
    print("This will collect top-selling books from 2019-2025 in each genre.")
    print("=" * 60)
    
    # Collect books for each genre
    collector.collect_horror_books(500)
    collector.collect_thriller_books(500)
    collector.collect_dystopian_books(500)
    collector.collect_fiction_books(500)
    collector.collect_romance_books(500)
    
    # Save all collections
    summary = collector.save_collections()
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE!")
    print(f"Total books collected: {summary['total_books']}")
    print("\nBy genre:")
    for genre, count in summary['by_genre'].items():
        print(f"  {genre}: {count} books")
    
    return collector

if __name__ == "__main__":
    collector = main()