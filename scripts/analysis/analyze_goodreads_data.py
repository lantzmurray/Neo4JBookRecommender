import pandas as pd
import json
from collections import defaultdict, Counter

def analyze_goodreads_export():
    """Analyze the Goodreads export to understand user preferences and extract all books."""
    
    # Read the CSV file
    df = pd.read_csv('goodreads_library_export.csv')
    
    print(f"Total books in Goodreads export: {len(df)}")
    print(f"Columns available: {list(df.columns)}")
    
    # Clean and analyze the data
    df['Year Published'] = pd.to_numeric(df['Year Published'], errors='coerce')
    df['Average Rating'] = pd.to_numeric(df['Average Rating'], errors='coerce')
    df['My Rating'] = pd.to_numeric(df['My Rating'], errors='coerce')
    
    # Filter for recent books (2019-2025)
    recent_books = df[df['Year Published'] >= 2019]
    print(f"Books from 2019-2025: {len(recent_books)}")
    
    # Analyze genres based on title keywords and bookshelves
    horror_keywords = ['horror', 'ghost', 'haunted', 'terror', 'nightmare', 'demon', 'zombie', 'vampire', 'werewolf', 'monster', 'scary', 'creepy', 'dark', 'evil', 'blood', 'death', 'kill', 'murder']
    thriller_keywords = ['thriller', 'suspense', 'psychological', 'mystery', 'crime', 'detective', 'investigation', 'murder', 'killer', 'dangerous', 'chase', 'escape']
    dystopian_keywords = ['dystopian', 'apocalypse', 'post-apocalyptic', 'future', 'society', 'rebellion', 'totalitarian', 'oppression', 'survival']
    romance_keywords = ['romance', 'love', 'wedding', 'marriage', 'relationship', 'dating', 'boyfriend', 'girlfriend', 'husband', 'wife']
    
    def classify_genre(title, bookshelves=''):
        """Classify book genre based on title and bookshelves."""
        title_lower = str(title).lower()
        shelves_lower = str(bookshelves).lower()
        combined = f"{title_lower} {shelves_lower}"
        
        genres = []
        
        # Check for horror
        if any(keyword in combined for keyword in horror_keywords):
            genres.append('Horror')
        
        # Check for thriller/suspense
        if any(keyword in combined for keyword in thriller_keywords):
            genres.append('Thriller')
        
        # Check for dystopian
        if any(keyword in combined for keyword in dystopian_keywords):
            genres.append('Dystopian')
        
        # Check for romance
        if any(keyword in combined for keyword in romance_keywords):
            genres.append('Romance')
        
        # Default to Fiction if no specific genre found
        if not genres:
            genres.append('Fiction')
        
        return genres
    
    # Classify all books
    df['Genres'] = df.apply(lambda row: classify_genre(row['Title'], row.get('Bookshelves', '')), axis=1)
    
    # Count books by genre
    genre_counts = Counter()
    for genres_list in df['Genres']:
        for genre in genres_list:
            genre_counts[genre] += 1
    
    print("\nGenre distribution in Goodreads library:")
    for genre, count in genre_counts.most_common():
        print(f"  {genre}: {count} books")
    
    # Extract high-rated books (4+ stars average)
    high_rated = df[df['Average Rating'] >= 4.0]
    print(f"\nHigh-rated books (4+ stars): {len(high_rated)}")
    
    # Extract user's favorites (5-star ratings)
    user_favorites = df[df['My Rating'] == 5]
    print(f"User's 5-star books: {len(user_favorites)}")
    
    # Prepare data for book collection
    goodreads_books = []
    
    for _, row in df.iterrows():
        book_data = {
            'goodreads_id': row['Book Id'],
            'title': row['Title'],
            'author': row['Author'],
            'isbn': str(row.get('ISBN', '')).replace('="', '').replace('"', ''),
            'isbn13': str(row.get('ISBN13', '')).replace('="', '').replace('"', ''),
            'average_rating': row['Average Rating'],
            'user_rating': row['My Rating'],
            'publisher': row.get('Publisher', ''),
            'year_published': row['Year Published'],
            'pages': row.get('Number of Pages', 0),
            'genres': row['Genres'],
            'read_status': row.get('Exclusive Shelf', ''),
            'date_read': row.get('Date Read', ''),
            'source': 'goodreads_export'
        }
        goodreads_books.append(book_data)
    
    # Save the analyzed data
    with open('goodreads_analyzed_books.json', 'w', encoding='utf-8') as f:
        json.dump(goodreads_books, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nSaved {len(goodreads_books)} books to goodreads_analyzed_books.json")
    
    # Show some examples of each genre
    print("\nExample books by genre:")
    for genre in ['Horror', 'Thriller', 'Dystopian', 'Romance', 'Fiction']:
        genre_books = [book for book in goodreads_books if genre in book['genres']]
        if genre_books:
            print(f"\n{genre} ({len(genre_books)} books):")
            for book in genre_books[:3]:  # Show first 3 examples
                print(f"  - {book['title']} by {book['author']} ({book['year_published']})")
    
    return goodreads_books, genre_counts

if __name__ == "__main__":
    books, genres = analyze_goodreads_export()