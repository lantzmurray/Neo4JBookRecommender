import json
import pandas as pd
from collections import defaultdict, Counter
import re
from datetime import datetime

class DatasetMerger:
    def __init__(self):
        self.goodreads_books = []
        self.collected_books = []
        self.final_dataset = []
        self.genre_targets = {
            'Horror': 500,
            'Thriller': 500,
            'Dystopian': 500,
            'Fiction': 500,
            'Romance': 500
        }
    
    def load_goodreads_books(self):
        """Load the analyzed Goodreads books."""
        try:
            with open('goodreads_analyzed_books.json', 'r', encoding='utf-8') as f:
                self.goodreads_books = json.load(f)
            print(f"Loaded {len(self.goodreads_books)} books from Goodreads export")
            return True
        except FileNotFoundError:
            print("Goodreads analyzed books file not found. Please run analyze_goodreads_data.py first.")
            return False
    
    def load_collected_books(self):
        """Load all collected books from the collection process."""
        collected_files = [
            'collected_horror_books.json',
            'collected_thriller_books.json',
            'collected_dystopian_books.json',
            'collected_fiction_books.json',
            'collected_romance_books.json'
        ]
        
        all_collected = []
        for filename in collected_files:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    books = json.load(f)
                    all_collected.extend(books)
                    print(f"Loaded {len(books)} books from {filename}")
            except FileNotFoundError:
                print(f"File {filename} not found - collection may still be in progress")
        
        # Also try to load from the combined file
        try:
            with open('all_collected_books.json', 'r', encoding='utf-8') as f:
                combined_books = json.load(f)
                if len(combined_books) > len(all_collected):
                    all_collected = combined_books
                    print(f"Loaded {len(combined_books)} books from combined collection file")
        except FileNotFoundError:
            pass
        
        self.collected_books = all_collected
        print(f"Total collected books loaded: {len(self.collected_books)}")
        return len(self.collected_books) > 0
    
    def normalize_book_data(self, book, source_type):
        """Normalize book data to a consistent format."""
        normalized = {
            'title': str(book.get('title', '')).strip(),
            'author': str(book.get('author', '')).strip(),
            'isbn': str(book.get('isbn', '')).strip(),
            'isbn13': str(book.get('isbn13', '')).strip(),
            'year_published': book.get('year_published'),
            'publisher': str(book.get('publisher', '')).strip(),
            'pages': book.get('pages', 0),
            'average_rating': book.get('average_rating', 0),
            'description': str(book.get('description', '')).strip(),
            'genres': book.get('genres', []),
            'source': source_type
        }
        
        # Ensure genres is a list
        if isinstance(normalized['genres'], str):
            normalized['genres'] = [normalized['genres']]
        
        # Clean up year
        if normalized['year_published']:
            try:
                normalized['year_published'] = int(normalized['year_published'])
            except:
                normalized['year_published'] = None
        
        # Clean up rating
        try:
            normalized['average_rating'] = float(normalized['average_rating'])
        except:
            normalized['average_rating'] = 0.0
        
        # Clean up pages
        try:
            normalized['pages'] = int(normalized['pages'])
        except:
            normalized['pages'] = 0
        
        # Add additional fields for recommendation system
        normalized.update({
            'goodreads_id': book.get('goodreads_id', ''),
            'google_id': book.get('google_id', ''),
            'user_rating': book.get('user_rating', 0),
            'ratings_count': book.get('ratings_count', 0),
            'categories': book.get('categories', []),
            'read_status': book.get('read_status', ''),
            'date_read': book.get('date_read', ''),
            'popularity_score': 0,  # Will be calculated
            'recommendation_features': {}  # Will be populated
        })
        
        return normalized
    
    def is_duplicate(self, book1, book2):
        """Check if two books are duplicates."""
        # Normalize titles for comparison
        title1 = re.sub(r'[^\w\s]', '', book1['title'].lower()).strip()
        title2 = re.sub(r'[^\w\s]', '', book2['title'].lower()).strip()
        
        author1 = book1['author'].lower().strip()
        author2 = book2['author'].lower().strip()
        
        # Check by title and author similarity
        if title1 == title2 and author1 == author2:
            return True
        
        # Check by ISBN
        if (book1['isbn'] and book2['isbn'] and 
            book1['isbn'] == book2['isbn']):
            return True
        
        if (book1['isbn13'] and book2['isbn13'] and 
            book1['isbn13'] == book2['isbn13']):
            return True
        
        # Check by Goodreads ID
        if (book1['goodreads_id'] and book2['goodreads_id'] and 
            book1['goodreads_id'] == book2['goodreads_id']):
            return True
        
        return False
    
    def merge_duplicate_books(self, book1, book2):
        """Merge information from duplicate books, preferring Goodreads data."""
        merged = book1.copy()
        
        # Prefer Goodreads source for user-specific data
        if book2['source'] == 'goodreads_export':
            merged.update({
                'user_rating': book2['user_rating'],
                'read_status': book2['read_status'],
                'date_read': book2['date_read'],
                'goodreads_id': book2['goodreads_id']
            })
        
        # Merge genres
        all_genres = list(set(book1['genres'] + book2['genres']))
        merged['genres'] = all_genres
        
        # Use better rating data
        if book2['average_rating'] > book1['average_rating']:
            merged['average_rating'] = book2['average_rating']
            merged['ratings_count'] = book2['ratings_count']
        
        # Use more complete metadata
        for field in ['isbn', 'isbn13', 'publisher', 'description']:
            if not merged[field] and book2[field]:
                merged[field] = book2[field]
        
        # Combine sources
        sources = [book1['source'], book2['source']]
        merged['source'] = ', '.join(list(set(sources)))
        
        return merged
    
    def calculate_popularity_score(self, book):
        """Calculate a popularity score for ranking books."""
        score = 0
        
        # Rating component (0-50 points)
        if book['average_rating'] > 0:
            score += book['average_rating'] * 10
        
        # Ratings count component (0-30 points)
        if book['ratings_count'] > 0:
            # Logarithmic scale for ratings count
            import math
            score += min(30, math.log10(book['ratings_count']) * 10)
        
        # Recency bonus (0-20 points)
        if book['year_published']:
            current_year = datetime.now().year
            years_old = current_year - book['year_published']
            if years_old <= 5:  # Books from last 5 years
                score += 20 - (years_old * 3)
        
        # User rating bonus for Goodreads books
        if book['user_rating'] > 0:
            score += book['user_rating'] * 5
        
        book['popularity_score'] = score
        return score
    
    def balance_genres(self):
        """Balance the dataset to meet genre targets."""
        # Group books by primary genre
        books_by_genre = defaultdict(list)
        
        for book in self.final_dataset:
            primary_genre = book['genres'][0] if book['genres'] else 'Fiction'
            books_by_genre[primary_genre].append(book)
        
        # Sort each genre by popularity score
        for genre in books_by_genre:
            books_by_genre[genre].sort(key=lambda x: x['popularity_score'], reverse=True)
        
        # Balance to targets
        balanced_dataset = []
        
        for genre, target_count in self.genre_targets.items():
            available_books = books_by_genre.get(genre, [])
            
            if len(available_books) >= target_count:
                # Take top books
                selected_books = available_books[:target_count]
            else:
                # Take all available books
                selected_books = available_books
                
                # Fill remaining slots with Fiction books if needed
                if genre != 'Fiction':
                    remaining_needed = target_count - len(selected_books)
                    fiction_books = books_by_genre.get('Fiction', [])
                    
                    # Find fiction books not already selected
                    selected_titles = {book['title'] for book in balanced_dataset}
                    available_fiction = [book for book in fiction_books 
                                       if book['title'] not in selected_titles]
                    
                    additional_books = available_fiction[:remaining_needed]
                    selected_books.extend(additional_books)
            
            balanced_dataset.extend(selected_books)
            print(f"Selected {len(selected_books)} books for {genre} genre")
        
        self.final_dataset = balanced_dataset
        return len(balanced_dataset)
    
    def enrich_for_recommendations(self):
        """Enrich book data with features for the recommendation system."""
        for book in self.final_dataset:
            features = {}
            
            # Extract features from description
            description = book['description'].lower()
            
            # Theme features
            themes = {
                'dark': any(word in description for word in ['dark', 'darkness', 'shadow', 'evil']),
                'romantic': any(word in description for word in ['love', 'romance', 'relationship', 'heart']),
                'action': any(word in description for word in ['action', 'fight', 'battle', 'war', 'chase']),
                'mystery': any(word in description for word in ['mystery', 'secret', 'hidden', 'discover']),
                'family': any(word in description for word in ['family', 'mother', 'father', 'daughter', 'son']),
                'supernatural': any(word in description for word in ['magic', 'supernatural', 'ghost', 'spirit']),
                'psychological': any(word in description for word in ['mind', 'psychological', 'mental', 'memory']),
                'historical': any(word in description for word in ['historical', 'history', 'past', 'century'])
            }
            
            # Setting features
            settings = {
                'contemporary': book['year_published'] and book['year_published'] >= 2000,
                'recent': book['year_published'] and book['year_published'] >= 2019,
                'classic': book['year_published'] and book['year_published'] < 1980,
                'long_book': book['pages'] > 400,
                'short_book': book['pages'] < 250
            }
            
            # Quality indicators
            quality = {
                'highly_rated': book['average_rating'] >= 4.0,
                'popular': book['ratings_count'] > 1000,
                'user_favorite': book['user_rating'] >= 4,
                'award_potential': book['average_rating'] >= 4.2 and book['ratings_count'] > 500
            }
            
            features.update(themes)
            features.update(settings)
            features.update(quality)
            
            book['recommendation_features'] = features
    
    def save_final_dataset(self):
        """Save the final merged and balanced dataset."""
        # Save the complete dataset
        with open('final_book_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(self.final_dataset, f, indent=2, ensure_ascii=False, default=str)
        
        # Create a CSV version for easy viewing
        df = pd.DataFrame(self.final_dataset)
        df.to_csv('final_book_dataset.csv', index=False, encoding='utf-8')
        
        # Create summary statistics
        summary = {
            'total_books': len(self.final_dataset),
            'by_genre': {},
            'by_source': {},
            'year_range': {},
            'rating_stats': {},
            'creation_date': datetime.now().isoformat()
        }
        
        # Genre distribution
        genre_counts = Counter()
        for book in self.final_dataset:
            for genre in book['genres']:
                genre_counts[genre] += 1
        summary['by_genre'] = dict(genre_counts)
        
        # Source distribution
        source_counts = Counter(book['source'] for book in self.final_dataset)
        summary['by_source'] = dict(source_counts)
        
        # Year distribution
        years = [book['year_published'] for book in self.final_dataset if book['year_published']]
        if years:
            summary['year_range'] = {
                'min_year': min(years),
                'max_year': max(years),
                'avg_year': sum(years) / len(years)
            }
        
        # Rating statistics
        ratings = [book['average_rating'] for book in self.final_dataset if book['average_rating'] > 0]
        if ratings:
            summary['rating_stats'] = {
                'avg_rating': sum(ratings) / len(ratings),
                'min_rating': min(ratings),
                'max_rating': max(ratings),
                'books_with_ratings': len(ratings)
            }
        
        with open('dataset_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nFinal dataset saved:")
        print(f"  - {len(self.final_dataset)} books total")
        print(f"  - JSON: final_book_dataset.json")
        print(f"  - CSV: final_book_dataset.csv")
        print(f"  - Summary: dataset_summary.json")
        
        return summary
    
    def merge_all_data(self):
        """Main method to merge all data sources."""
        print("Starting dataset merge and preparation...")
        
        # Load data
        if not self.load_goodreads_books():
            return False
        
        self.load_collected_books()  # This may be empty if collection is still running
        
        # Normalize all books
        all_books = []
        
        # Add Goodreads books
        for book in self.goodreads_books:
            normalized = self.normalize_book_data(book, 'goodreads_export')
            all_books.append(normalized)
        
        # Add collected books
        for book in self.collected_books:
            normalized = self.normalize_book_data(book, 'google_books')
            all_books.append(normalized)
        
        print(f"Total books before deduplication: {len(all_books)}")
        
        # Remove duplicates and merge information
        unique_books = []
        for book in all_books:
            # Check if this book is a duplicate of any existing book
            duplicate_found = False
            for i, existing_book in enumerate(unique_books):
                if self.is_duplicate(book, existing_book):
                    # Merge the duplicate
                    unique_books[i] = self.merge_duplicate_books(existing_book, book)
                    duplicate_found = True
                    break
            
            if not duplicate_found:
                unique_books.append(book)
        
        print(f"Unique books after deduplication: {len(unique_books)}")
        
        # Calculate popularity scores
        for book in unique_books:
            self.calculate_popularity_score(book)
        
        self.final_dataset = unique_books
        
        # Balance genres if we have enough books
        if len(unique_books) >= 1000:
            self.balance_genres()
        
        # Enrich for recommendations
        self.enrich_for_recommendations()
        
        # Save final dataset
        summary = self.save_final_dataset()
        
        return summary

def main():
    merger = DatasetMerger()
    summary = merger.merge_all_data()
    
    if summary:
        print("\n" + "=" * 60)
        print("DATASET PREPARATION COMPLETE!")
        print(f"Final dataset contains {summary['total_books']} books")
        print("\nGenre distribution:")
        for genre, count in summary['by_genre'].items():
            print(f"  {genre}: {count} books")
        print("\nSource distribution:")
        for source, count in summary['by_source'].items():
            print(f"  {source}: {count} books")
        
        if 'rating_stats' in summary:
            print(f"\nAverage rating: {summary['rating_stats']['avg_rating']:.2f}")
        
        return merger
    else:
        print("Failed to create dataset. Please ensure all required files are available.")
        return None

if __name__ == "__main__":
    merger = main()