# 📚 API Documentation

This document provides comprehensive documentation for the Book Recommendation System's internal APIs and data processing pipeline.

## 🏗️ System Architecture

The Book Recommendation System consists of several interconnected components:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Processing │───▶│   Neo4j Graph   │
│   (Web APIs)    │    │    Pipeline      │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   OpenAI API     │    │   Streamlit     │
                       │   (Embeddings)   │    │   Web App       │
                       └──────────────────┘    └─────────────────┘
```

## 🔌 Core APIs and Modules

### 1. Data Collection APIs

#### BookCollector Class
**File**: `collect_*_books.py` (e.g., `collect_horror_books.py`)

```python
class BookCollector:
    def __init__(self, base_url: str, headers: dict = None)
    def collect_books(self, genre: str, max_pages: int = 50) -> List[Dict]
    def fetch_book_details(self, book_id: str) -> Dict
    def save_to_json(self, books: List[Dict], filename: str) -> None
```

**Methods**:

##### `collect_books(genre, max_pages)`
Collects books from external APIs by genre.

**Parameters**:
- `genre` (str): Book genre to collect
- `max_pages` (int): Maximum pages to scrape (default: 50)

**Returns**: List of book dictionaries

**Example**:
```python
collector = BookCollector("https://api.example.com")
horror_books = collector.collect_books("horror", max_pages=10)
```

##### `fetch_book_details(book_id)`
Fetches detailed information for a specific book.

**Parameters**:
- `book_id` (str): Unique book identifier

**Returns**: Dictionary with book details

**Example Response**:
```json
{
    "id": "12345",
    "title": "The Haunting of Hill House",
    "author": "Shirley Jackson",
    "isbn": "978-0143039983",
    "publication_year": 1959,
    "genre": ["Horror", "Gothic Fiction"],
    "rating": 4.1,
    "description": "A classic horror novel...",
    "page_count": 246,
    "publisher": "Penguin Classics"
}
```

### 2. Data Processing Pipeline

#### DataMerger Class
**File**: `merge_all_collections.py`

```python
class DataMerger:
    def __init__(self, input_directory: str = ".")
    def load_collections(self) -> List[Dict]
    def deduplicate_books(self, books: List[Dict]) -> List[Dict]
    def normalize_data(self, books: List[Dict]) -> List[Dict]
    def merge_and_save(self, output_file: str = "merged_book_dataset.json") -> Dict
```

**Methods**:

##### `load_collections()`
Loads all book collection JSON files from the directory.

**Returns**: List of all books from all collections

##### `deduplicate_books(books)`
Removes duplicate books based on ISBN, title, and author matching.

**Parameters**:
- `books` (List[Dict]): List of book dictionaries

**Returns**: Deduplicated list of books

**Deduplication Logic**:
1. Primary: ISBN matching
2. Secondary: Title + Author similarity (>90%)
3. Tertiary: Title similarity + publication year

##### `normalize_data(books)`
Standardizes book data format and cleans inconsistencies.

**Normalization Steps**:
- Standardize genre names
- Clean publication years
- Normalize author names
- Validate ISBNs
- Clean descriptions

**Example Usage**:
```python
merger = DataMerger()
books = merger.load_collections()
clean_books = merger.normalize_data(books)
summary = merger.merge_and_save("merged_dataset.json")
```

### 3. Embedding Generation API

#### BookVectorizer Class
**File**: `vectorize_books.py`

```python
class BookVectorizer:
    def __init__(self, openai_api_key: str = None)
    def create_text_representation(self, book: Dict) -> str
    def generate_embeddings(self, books: List[Dict], batch_size: int = 100) -> List[Dict]
    def calculate_similarities(self, embeddings: np.ndarray) -> Dict
    def save_results(self, books_with_embeddings: List[Dict], analysis: Dict) -> None
```

**Methods**:

##### `create_text_representation(book)`
Creates a comprehensive text representation of a book for embedding generation.

**Parameters**:
- `book` (Dict): Book dictionary

**Returns**: Formatted text string

**Text Format**:
```
Title: {title}
Author: {author}
Genre: {genres}
Description: {description}
Publication Year: {year}
Publisher: {publisher}
```

##### `generate_embeddings(books, batch_size)`
Generates vector embeddings using OpenAI's text-embedding-ada-002 model.

**Parameters**:
- `books` (List[Dict]): List of books
- `batch_size` (int): Processing batch size (default: 100)

**Returns**: Books with added embedding vectors

**Rate Limiting**: Automatically handles OpenAI API rate limits with exponential backoff.

**Example**:
```python
vectorizer = BookVectorizer(api_key="sk-...")
books_with_embeddings = vectorizer.generate_embeddings(books)
```

##### `calculate_similarities(embeddings)`
Calculates cosine similarity between all book pairs.

**Parameters**:
- `embeddings` (np.ndarray): Array of embedding vectors

**Returns**: Dictionary with similarity analysis

**Analysis Includes**:
- Total comparisons made
- High similarity pairs (>0.7)
- Mean similarity score
- Top similar book pairs

### 4. Neo4j Database API

#### Neo4jBookUploader Class
**File**: `upload_to_neo4j.py`

```python
class Neo4jBookUploader:
    def __init__(self, uri: str, user: str, password: str)
    def verify_connection(self) -> bool
    def create_indexes(self) -> None
    def upload_books(self, books: List[Dict]) -> Dict
    def create_relationships(self, books: List[Dict]) -> Dict
    def get_upload_summary(self) -> Dict
```

**Methods**:

##### `create_indexes()`
Creates database indexes for optimal query performance.

**Indexes Created**:
```cypher
CREATE INDEX book_id_index FOR (b:Book) ON (b.id)
CREATE INDEX book_title_index FOR (b:Book) ON (b.title)
CREATE INDEX book_isbn_index FOR (b:Book) ON (b.isbn)
CREATE INDEX author_name_index FOR (a:Author) ON (a.name)
CREATE INDEX genre_name_index FOR (g:Genre) ON (g.name)
CREATE INDEX publisher_name_index FOR (p:Publisher) ON (p.name)
```

##### `upload_books(books)`
Uploads books and creates nodes in Neo4j.

**Node Types Created**:
- `Book`: Main book entities
- `Author`: Book authors
- `Genre`: Book genres
- `Publisher`: Publishing companies

**Example Cypher Query**:
```cypher
MERGE (b:Book {id: $book_id})
SET b.title = $title,
    b.isbn = $isbn,
    b.publication_year = $year,
    b.rating = $rating,
    b.description = $description,
    b.embedding = $embedding
```

##### `create_relationships(books)`
Creates relationships between nodes.

**Relationship Types**:
- `WROTE`: Author → Book
- `BELONGS_TO`: Book → Genre
- `PUBLISHED`: Publisher → Book
- `SIMILAR_TO`: Book → Book (based on embeddings)

### 5. Web Application API

#### BookRecommendationSystem Class
**File**: `streamlit_app.py`

```python
class BookRecommendationSystem:
    def __init__(self)
    def initialize_neo4j(self) -> bool
    def load_books_data(self) -> bool
    def search_books(self, query: str, limit: int = 10) -> List[Dict]
    def get_similar_books(self, book_id: str, limit: int = 5) -> List[Dict]
    def get_books_by_genre(self, genre: str, limit: int = 20) -> List[Dict]
    def get_book_details(self, book_id: str) -> Dict
```

**Methods**:

##### `search_books(query, limit)`
Searches books by title, author, or description.

**Parameters**:
- `query` (str): Search query
- `limit` (int): Maximum results (default: 10)

**Returns**: List of matching books

**Cypher Query**:
```cypher
MATCH (b:Book)
WHERE toLower(b.title) CONTAINS toLower($query)
   OR toLower(b.author) CONTAINS toLower($query)
   OR toLower(b.description) CONTAINS toLower($query)
RETURN b
LIMIT $limit
```

##### `get_similar_books(book_id, limit)`
Finds books similar to a given book using embedding similarity.

**Parameters**:
- `book_id` (str): Target book ID
- `limit` (int): Number of recommendations (default: 5)

**Returns**: List of similar books with similarity scores

**Algorithm**:
1. Retrieve target book's embedding
2. Calculate cosine similarity with all other books
3. Return top N most similar books

##### `get_books_by_genre(genre, limit)`
Retrieves books filtered by genre.

**Parameters**:
- `genre` (str): Genre name
- `limit` (int): Maximum results (default: 20)

**Returns**: List of books in the specified genre

## 🔍 Query Examples

### Neo4j Cypher Queries

#### Find Books by Author
```cypher
MATCH (a:Author)-[:WROTE]->(b:Book)
WHERE a.name CONTAINS "Stephen King"
RETURN b.title, b.publication_year, b.rating
ORDER BY b.rating DESC
```

#### Get Genre Statistics
```cypher
MATCH (b:Book)-[:BELONGS_TO]->(g:Genre)
RETURN g.name as genre, 
       count(b) as book_count,
       avg(b.rating) as avg_rating
ORDER BY book_count DESC
```

#### Find Similar Books
```cypher
MATCH (b1:Book)-[s:SIMILAR_TO]->(b2:Book)
WHERE b1.id = $book_id
RETURN b2.title, b2.author, s.similarity_score
ORDER BY s.similarity_score DESC
LIMIT 5
```

#### Publisher Analysis
```cypher
MATCH (p:Publisher)-[:PUBLISHED]->(b:Book)
RETURN p.name as publisher,
       count(b) as books_published,
       avg(b.rating) as avg_rating,
       min(b.publication_year) as earliest_year,
       max(b.publication_year) as latest_year
ORDER BY books_published DESC
```

### Python API Usage Examples

#### Complete Data Processing Pipeline
```python
# 1. Collect books
from collect_horror_books import collect_horror_books
horror_books = collect_horror_books(max_pages=10)

# 2. Merge collections
from merge_all_collections import DataMerger
merger = DataMerger()
merged_data = merger.merge_and_save()

# 3. Generate embeddings
from vectorize_books import BookVectorizer
vectorizer = BookVectorizer()
books_with_embeddings = vectorizer.generate_embeddings(merged_data['books'])

# 4. Upload to Neo4j
from upload_to_neo4j import Neo4jBookUploader
uploader = Neo4jBookUploader("bolt://localhost:7687", "neo4j", "password")
upload_summary = uploader.upload_books(books_with_embeddings)
```

#### Search and Recommendation
```python
from streamlit_app import BookRecommendationSystem

# Initialize system
recommender = BookRecommendationSystem()
recommender.initialize_neo4j()
recommender.load_books_data()

# Search for books
results = recommender.search_books("harry potter")
print(f"Found {len(results)} books")

# Get recommendations
if results:
    book_id = results[0]['id']
    similar_books = recommender.get_similar_books(book_id, limit=5)
    print(f"Similar books: {[book['title'] for book in similar_books]}")
```

## 📊 Data Formats

### Book Data Structure
```json
{
    "id": "unique_book_id",
    "title": "Book Title",
    "author": "Author Name",
    "isbn": "978-1234567890",
    "publication_year": 2023,
    "genre": ["Fiction", "Mystery"],
    "rating": 4.2,
    "rating_count": 1500,
    "description": "Book description...",
    "page_count": 320,
    "publisher": "Publisher Name",
    "language": "English",
    "source": "goodreads",
    "embedding": [0.1, -0.2, 0.3, ...],  // 1536-dimensional vector
    "normalized_genres": ["fiction", "mystery"],
    "text_representation": "Title: Book Title\nAuthor: Author Name..."
}
```

### Embedding Analysis Structure
```json
{
    "total_books": 2131,
    "embedding_dimension": 1536,
    "model_used": "text-embedding-ada-002",
    "total_comparisons": 2267565,
    "high_similarity_pairs": 4941,
    "similarity_threshold": 0.7,
    "mean_similarity": 0.775,
    "processing_time_seconds": 1247.3,
    "top_similar_pairs": [
        {
            "book1_id": "id1",
            "book1_title": "Title 1",
            "book2_id": "id2", 
            "book2_title": "Title 2",
            "similarity_score": 0.95
        }
    ]
}
```

### Upload Summary Structure
```json
{
    "total_books": 2131,
    "books_uploaded": 2118,
    "total_authors": 1456,
    "total_genres": 23,
    "total_publishers": 541,
    "wrote_relationships": 2118,
    "belongs_to_relationships": 4236,
    "published_relationships": 2011,
    "similarity_relationships": 4941,
    "upload_time_seconds": 89.7,
    "errors": [],
    "warnings": ["Unknown property key: normalized_genres"]
}
```

## 🚨 Error Handling

### Common Error Codes

#### Data Collection Errors
- `RATE_LIMIT_EXCEEDED`: API rate limit reached
- `INVALID_RESPONSE`: Malformed API response
- `NETWORK_ERROR`: Connection timeout or failure
- `AUTHENTICATION_ERROR`: Invalid API credentials

#### Processing Errors
- `DUPLICATE_DETECTION_FAILED`: Error in deduplication logic
- `EMBEDDING_GENERATION_FAILED`: OpenAI API error
- `DATA_VALIDATION_ERROR`: Invalid book data format

#### Database Errors
- `NEO4J_CONNECTION_ERROR`: Cannot connect to Neo4j
- `CYPHER_QUERY_ERROR`: Invalid Cypher syntax
- `CONSTRAINT_VIOLATION`: Database constraint violation
- `TRANSACTION_FAILED`: Database transaction rollback

### Error Response Format
```json
{
    "error": true,
    "error_code": "EMBEDDING_GENERATION_FAILED",
    "message": "Failed to generate embeddings for batch",
    "details": {
        "batch_size": 100,
        "failed_books": ["book_id_1", "book_id_2"],
        "api_response": "Rate limit exceeded"
    },
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔧 Configuration Options

### Environment Variables
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=text-embedding-ada-002
OPENAI_MAX_RETRIES=3

# Processing Configuration
BATCH_SIZE=100
SIMILARITY_THRESHOLD=0.7
MAX_BOOKS_PER_GENRE=1000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=book_recommender.log
```

### Streamlit Configuration
```toml
# .streamlit/config.toml
[server]
port = 8501
address = "localhost"
maxUploadSize = 200

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[browser]
gatherUsageStats = false
```

## 📈 Performance Metrics

### Typical Processing Times
- **Book Collection**: ~2-5 minutes per 100 books
- **Data Merging**: ~30 seconds for 2000 books
- **Embedding Generation**: ~20 minutes for 2000 books
- **Neo4j Upload**: ~2 minutes for 2000 books
- **Similarity Calculation**: ~5 minutes for 2000 books

### Memory Usage
- **Data Processing**: ~500MB for 2000 books
- **Embedding Storage**: ~12MB for 2000 books (1536-dim)
- **Neo4j Database**: ~200MB for 2000 books with relationships

### API Rate Limits
- **OpenAI Embeddings**: 3,000 requests/minute (Tier 1)
- **Book APIs**: Varies by source (typically 100-1000/hour)
- **Neo4j**: No inherent limits (hardware dependent)

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Never commit keys to version control
- Use key rotation for production deployments
- Monitor API usage for anomalies

### Database Security
- Change default Neo4j passwords
- Use encrypted connections (bolt+s://)
- Implement proper access controls
- Regular security updates

### Data Privacy
- Anonymize user data where possible
- Implement data retention policies
- Comply with GDPR/CCPA requirements
- Secure data transmission

---

This API documentation provides comprehensive coverage of all system components. For additional examples and advanced usage patterns, refer to the individual module documentation and the main README.md file.