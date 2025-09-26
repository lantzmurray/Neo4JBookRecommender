# 📊 Data Pipeline Documentation

This document provides a comprehensive overview of the Book Recommendation System's data collection, processing, and embedding pipeline.

## 🔄 Pipeline Overview

The data pipeline consists of five main stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  1. Collection  │───▶│  2. Validation  │───▶│  3. Merging     │
│  (Web Scraping) │    │  & Cleaning     │    │  & Deduplication│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             ▼
│  5. Database    │◀───│  4. Embedding   │    ┌─────────────────┐
│  Upload (Neo4j) │    │  Generation     │    │  Normalized     │
└─────────────────┘    └─────────────────┘    │  Dataset        │
                                              └─────────────────┘
```

## 📥 Stage 1: Data Collection

### Data Sources

The system collects book data from multiple sources to ensure comprehensive coverage:

#### Primary Sources
1. **Goodreads API** (via web scraping)
   - Comprehensive book metadata
   - User ratings and reviews
   - Genre classifications
   - Author information

2. **Open Library API**
   - ISBN validation
   - Publication details
   - Alternative metadata

3. **Google Books API**
   - Additional book descriptions
   - Cover images
   - Publisher information

### Collection Scripts

#### Genre-Specific Collectors
Each genre has a dedicated collection script:

- `collect_horror_books.py` - Horror and thriller books
- `collect_mystery_books.py` - Mystery and detective fiction
- `collect_romance_books.py` - Romance novels
- `collect_scifi_books.py` - Science fiction
- `collect_fantasy_books.py` - Fantasy literature
- `collect_literary_books.py` - Literary fiction
- `collect_nonfiction_books.py` - Non-fiction works

#### Collection Process

```python
# Example: Horror Books Collection
def collect_horror_books(max_pages=50):
    """
    Collects horror books from multiple sources
    
    Process:
    1. Query genre-specific endpoints
    2. Extract book metadata
    3. Validate data quality
    4. Handle rate limiting
    5. Save to JSON files
    """
    
    collector = BookCollector(
        base_url="https://www.goodreads.com/shelf/show/horror",
        headers={"User-Agent": "BookRecommender/1.0"}
    )
    
    books = []
    for page in range(1, max_pages + 1):
        try:
            page_books = collector.fetch_page(page)
            books.extend(page_books)
            
            # Rate limiting
            time.sleep(random.uniform(1, 3))
            
        except RateLimitError:
            print(f"Rate limit reached at page {page}")
            time.sleep(60)  # Wait 1 minute
            continue
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue
    
    # Save collected data
    save_to_json(books, "horror_books.json")
    return books
```

### Data Quality Metrics

For each collection run, we track:

- **Total books collected**: Number of unique books found
- **Success rate**: Percentage of successful API calls
- **Data completeness**: Percentage of books with all required fields
- **Duplicate rate**: Percentage of duplicate books detected
- **Error rate**: Percentage of failed requests

#### Example Collection Summary
```json
{
    "collection_date": "2024-01-15T10:30:00Z",
    "genre": "horror",
    "total_books_collected": 1247,
    "pages_processed": 25,
    "success_rate": 0.94,
    "data_completeness": {
        "title": 1.0,
        "author": 0.98,
        "isbn": 0.76,
        "description": 0.89,
        "rating": 0.92,
        "publication_year": 0.85
    },
    "duplicate_rate": 0.12,
    "error_rate": 0.06,
    "processing_time_minutes": 45.2
}
```

## 🧹 Stage 2: Data Validation & Cleaning

### Validation Rules

#### Required Fields Validation
```python
REQUIRED_FIELDS = {
    'title': str,
    'author': str,
    'genre': list,
    'rating': float,
    'description': str
}

def validate_book(book_data):
    """Validates book data against required schema"""
    errors = []
    
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in book_data:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(book_data[field], expected_type):
            errors.append(f"Invalid type for {field}: expected {expected_type}")
    
    return errors
```

#### Data Quality Checks
1. **Title Validation**
   - Non-empty strings
   - Reasonable length (5-200 characters)
   - No HTML tags or special characters

2. **Author Validation**
   - Valid author name format
   - Handle multiple authors
   - Standardize name formatting

3. **ISBN Validation**
   - Valid ISBN-10 or ISBN-13 format
   - Checksum verification
   - Remove hyphens and spaces

4. **Rating Validation**
   - Numeric values between 0.0 and 5.0
   - Handle different rating scales
   - Convert to standardized 5-point scale

5. **Publication Year Validation**
   - Reasonable year range (1800-2030)
   - Handle different date formats
   - Extract year from full dates

### Data Cleaning Process

#### Text Normalization
```python
def clean_text_fields(book):
    """Normalizes text fields for consistency"""
    
    # Clean title
    book['title'] = re.sub(r'<[^>]+>', '', book['title'])  # Remove HTML
    book['title'] = re.sub(r'\s+', ' ', book['title']).strip()  # Normalize whitespace
    
    # Clean description
    book['description'] = clean_html(book['description'])
    book['description'] = truncate_text(book['description'], max_length=2000)
    
    # Standardize author names
    book['author'] = standardize_author_name(book['author'])
    
    return book
```

#### Genre Normalization
```python
GENRE_MAPPING = {
    'sci-fi': 'science fiction',
    'scifi': 'science fiction',
    'fantasy fiction': 'fantasy',
    'romantic fiction': 'romance',
    'detective fiction': 'mystery',
    'thriller': 'mystery',
    # ... more mappings
}

def normalize_genres(genres):
    """Standardizes genre names"""
    normalized = []
    for genre in genres:
        genre_lower = genre.lower().strip()
        normalized_genre = GENRE_MAPPING.get(genre_lower, genre_lower)
        if normalized_genre not in normalized:
            normalized.append(normalized_genre)
    return normalized
```

### Cleaning Statistics

After cleaning, we generate statistics:

```json
{
    "cleaning_summary": {
        "books_processed": 1247,
        "books_cleaned": 1189,
        "books_rejected": 58,
        "rejection_reasons": {
            "missing_title": 12,
            "missing_author": 8,
            "invalid_rating": 15,
            "duplicate_isbn": 23
        },
        "cleaning_actions": {
            "html_removed": 234,
            "whitespace_normalized": 1189,
            "genres_standardized": 892,
            "authors_standardized": 456
        }
    }
}
```

## 🔄 Stage 3: Merging & Deduplication

### Merging Process

The `merge_all_collections.py` script combines all genre-specific collections:

```python
def merge_collections():
    """Merges all book collections into a single dataset"""
    
    all_books = []
    collection_files = glob.glob("*_books.json")
    
    for file in collection_files:
        with open(file, 'r', encoding='utf-8') as f:
            books = json.load(f)
            
            # Add source information
            genre = file.replace('_books.json', '')
            for book in books:
                book['source_collection'] = genre
                
            all_books.extend(books)
    
    print(f"Loaded {len(all_books)} books from {len(collection_files)} collections")
    return all_books
```

### Deduplication Algorithm

#### Multi-Level Deduplication Strategy

1. **Level 1: ISBN Matching**
   ```python
   def deduplicate_by_isbn(books):
       """Remove duplicates based on ISBN"""
       seen_isbns = set()
       unique_books = []
       
       for book in books:
           isbn = book.get('isbn', '').replace('-', '').replace(' ', '')
           if isbn and isbn not in seen_isbns:
               seen_isbns.add(isbn)
               unique_books.append(book)
           elif not isbn:
               unique_books.append(book)  # Keep books without ISBN for further processing
       
       return unique_books
   ```

2. **Level 2: Title + Author Matching**
   ```python
   def deduplicate_by_title_author(books):
       """Remove duplicates based on title and author similarity"""
       unique_books = []
       
       for book in books:
           is_duplicate = False
           
           for existing_book in unique_books:
               title_similarity = calculate_similarity(book['title'], existing_book['title'])
               author_similarity = calculate_similarity(book['author'], existing_book['author'])
               
               if title_similarity > 0.9 and author_similarity > 0.8:
                   # Merge information from both books
                   merged_book = merge_book_data(existing_book, book)
                   unique_books[unique_books.index(existing_book)] = merged_book
                   is_duplicate = True
                   break
           
           if not is_duplicate:
               unique_books.append(book)
       
       return unique_books
   ```

3. **Level 3: Fuzzy Matching**
   ```python
   def fuzzy_deduplicate(books):
       """Advanced deduplication using fuzzy string matching"""
       from fuzzywuzzy import fuzz
       
       unique_books = []
       
       for book in books:
           is_duplicate = False
           
           for existing_book in unique_books:
               # Create comparison strings
               book_str = f"{book['title']} {book['author']} {book.get('publication_year', '')}"
               existing_str = f"{existing_book['title']} {existing_book['author']} {existing_book.get('publication_year', '')}"
               
               similarity = fuzz.ratio(book_str.lower(), existing_str.lower())
               
               if similarity > 85:  # 85% similarity threshold
                   # Keep the book with more complete data
                   if count_non_empty_fields(book) > count_non_empty_fields(existing_book):
                       unique_books[unique_books.index(existing_book)] = book
                   is_duplicate = True
                   break
           
           if not is_duplicate:
               unique_books.append(book)
       
       return unique_books
   ```

### Merge Statistics

```json
{
    "merge_summary": {
        "input_collections": 7,
        "total_books_before_merge": 8456,
        "duplicates_removed": {
            "isbn_duplicates": 1234,
            "title_author_duplicates": 567,
            "fuzzy_duplicates": 234
        },
        "total_books_after_merge": 2131,
        "deduplication_rate": 0.748,
        "genre_distribution": {
            "horror": 387,
            "mystery": 423,
            "romance": 356,
            "science fiction": 298,
            "fantasy": 334,
            "literary fiction": 201,
            "non-fiction": 132
        },
        "data_completeness_after_merge": {
            "title": 1.0,
            "author": 0.99,
            "isbn": 0.78,
            "description": 0.91,
            "rating": 0.94,
            "publication_year": 0.87,
            "genre": 1.0
        }
    }
}
```

## 🤖 Stage 4: Embedding Generation

### Text Representation Creation

Before generating embeddings, we create comprehensive text representations:

```python
def create_text_representation(book):
    """Creates a rich text representation for embedding generation"""
    
    components = []
    
    # Title (weighted heavily)
    if book.get('title'):
        components.append(f"Title: {book['title']}")
    
    # Author information
    if book.get('author'):
        components.append(f"Author: {book['author']}")
    
    # Genre information (important for recommendations)
    if book.get('genre'):
        genres = ', '.join(book['genre']) if isinstance(book['genre'], list) else book['genre']
        components.append(f"Genre: {genres}")
    
    # Description (main content)
    if book.get('description'):
        # Truncate description to avoid token limits
        description = book['description'][:1500]
        components.append(f"Description: {description}")
    
    # Publication context
    if book.get('publication_year'):
        components.append(f"Publication Year: {book['publication_year']}")
    
    if book.get('publisher'):
        components.append(f"Publisher: {book['publisher']}")
    
    # Additional metadata
    if book.get('rating'):
        components.append(f"Rating: {book['rating']}/5.0")
    
    if book.get('page_count'):
        components.append(f"Pages: {book['page_count']}")
    
    return '\n'.join(components)
```

### OpenAI Embedding Generation

```python
class EmbeddingGenerator:
    def __init__(self, api_key, model="text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.rate_limiter = RateLimiter(requests_per_minute=3000)
    
    def generate_embeddings(self, books, batch_size=100):
        """Generates embeddings for a list of books"""
        
        books_with_embeddings = []
        total_batches = (len(books) + batch_size - 1) // batch_size
        
        for i in range(0, len(books), batch_size):
            batch = books[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} books)")
            
            try:
                # Create text representations
                texts = [create_text_representation(book) for book in batch]
                
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Generate embeddings
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                
                # Add embeddings to books
                for j, book in enumerate(batch):
                    book_copy = book.copy()
                    book_copy['embedding'] = response.data[j].embedding
                    book_copy['text_representation'] = texts[j]
                    books_with_embeddings.append(book_copy)
                
                # Progress tracking
                print(f"✅ Generated embeddings for {len(batch)} books")
                
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                # Retry the batch
                continue
                
            except Exception as e:
                print(f"Error processing batch {batch_num}: {e}")
                # Add books without embeddings to maintain data integrity
                for book in batch:
                    book_copy = book.copy()
                    book_copy['embedding'] = None
                    book_copy['embedding_error'] = str(e)
                    books_with_embeddings.append(book_copy)
        
        return books_with_embeddings
```

### Embedding Analysis

After generation, we analyze the embedding space:

```python
def analyze_embeddings(books_with_embeddings):
    """Analyzes the generated embeddings"""
    
    # Extract embeddings
    embeddings = []
    valid_books = []
    
    for book in books_with_embeddings:
        if book.get('embedding'):
            embeddings.append(book['embedding'])
            valid_books.append(book)
    
    embeddings = np.array(embeddings)
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    # Find high-similarity pairs
    high_similarity_pairs = []
    threshold = 0.7
    
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            similarity = similarities[i][j]
            if similarity > threshold:
                high_similarity_pairs.append({
                    'book1_id': valid_books[i]['id'],
                    'book1_title': valid_books[i]['title'],
                    'book2_id': valid_books[j]['id'],
                    'book2_title': valid_books[j]['title'],
                    'similarity_score': float(similarity)
                })
    
    # Sort by similarity score
    high_similarity_pairs.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    analysis = {
        'total_books': len(books_with_embeddings),
        'books_with_embeddings': len(valid_books),
        'embedding_dimension': len(embeddings[0]) if len(embeddings) > 0 else 0,
        'model_used': 'text-embedding-ada-002',
        'total_comparisons': len(similarities) * (len(similarities) - 1) // 2,
        'high_similarity_pairs': len(high_similarity_pairs),
        'similarity_threshold': threshold,
        'mean_similarity': float(np.mean(similarities[np.triu_indices_from(similarities, k=1)])),
        'top_similar_pairs': high_similarity_pairs[:50]  # Top 50 most similar pairs
    }
    
    return analysis
```

### Embedding Quality Metrics

```json
{
    "embedding_analysis": {
        "total_books": 2131,
        "books_with_embeddings": 2131,
        "embedding_dimension": 1536,
        "model_used": "text-embedding-ada-002",
        "generation_time_minutes": 23.4,
        "api_calls_made": 22,
        "tokens_processed": 1247832,
        "cost_estimate_usd": 0.12,
        "quality_metrics": {
            "mean_similarity": 0.234,
            "std_similarity": 0.156,
            "high_similarity_pairs": 4941,
            "similarity_threshold": 0.7,
            "genre_coherence": 0.78,
            "author_coherence": 0.82
        },
        "error_rate": 0.0,
        "failed_embeddings": 0
    }
}
```

## 💾 Stage 5: Database Upload

### Neo4j Schema Design

```cypher
-- Node Types
CREATE CONSTRAINT book_id_unique FOR (b:Book) REQUIRE b.id IS UNIQUE;
CREATE CONSTRAINT author_name_unique FOR (a:Author) REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT genre_name_unique FOR (g:Genre) REQUIRE g.name IS UNIQUE;
CREATE CONSTRAINT publisher_name_unique FOR (p:Publisher) REQUIRE p.name IS UNIQUE;

-- Indexes for Performance
CREATE INDEX book_title_index FOR (b:Book) ON (b.title);
CREATE INDEX book_rating_index FOR (b:Book) ON (b.rating);
CREATE INDEX book_year_index FOR (b:Book) ON (b.publication_year);
CREATE INDEX author_name_index FOR (a:Author) ON (a.name);
```

### Upload Process

```python
class Neo4jUploader:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def upload_books(self, books_with_embeddings):
        """Uploads books and creates relationships"""
        
        with self.driver.session() as session:
            # Create books
            for book in books_with_embeddings:
                session.execute_write(self._create_book, book)
            
            # Create relationships
            for book in books_with_embeddings:
                session.execute_write(self._create_relationships, book)
            
            # Create similarity relationships
            self._create_similarity_relationships(books_with_embeddings)
    
    def _create_book(self, tx, book):
        """Creates a book node with all properties"""
        
        query = """
        MERGE (b:Book {id: $id})
        SET b.title = $title,
            b.author = $author,
            b.isbn = $isbn,
            b.publication_year = $publication_year,
            b.rating = $rating,
            b.rating_count = $rating_count,
            b.description = $description,
            b.page_count = $page_count,
            b.language = $language,
            b.source = $source,
            b.embedding = $embedding,
            b.updated_at = datetime()
        """
        
        tx.run(query, **book)
    
    def _create_relationships(self, tx, book):
        """Creates relationships between book and other entities"""
        
        # Author relationship
        if book.get('author'):
            tx.run("""
                MATCH (b:Book {id: $book_id})
                MERGE (a:Author {name: $author_name})
                MERGE (a)-[:WROTE]->(b)
            """, book_id=book['id'], author_name=book['author'])
        
        # Genre relationships
        if book.get('genre'):
            genres = book['genre'] if isinstance(book['genre'], list) else [book['genre']]
            for genre in genres:
                tx.run("""
                    MATCH (b:Book {id: $book_id})
                    MERGE (g:Genre {name: $genre_name})
                    MERGE (b)-[:BELONGS_TO]->(g)
                """, book_id=book['id'], genre_name=genre.lower())
        
        # Publisher relationship
        if book.get('publisher'):
            tx.run("""
                MATCH (b:Book {id: $book_id})
                MERGE (p:Publisher {name: $publisher_name})
                MERGE (p)-[:PUBLISHED]->(b)
            """, book_id=book['id'], publisher_name=book['publisher'])
    
    def _create_similarity_relationships(self, books_with_embeddings):
        """Creates similarity relationships based on embeddings"""
        
        # Calculate similarities and create relationships
        embeddings = [book['embedding'] for book in books_with_embeddings if book.get('embedding')]
        similarities = cosine_similarity(embeddings)
        
        threshold = 0.7
        relationships_created = 0
        
        with self.driver.session() as session:
            for i in range(len(similarities)):
                for j in range(i + 1, len(similarities)):
                    similarity = similarities[i][j]
                    
                    if similarity > threshold:
                        session.run("""
                            MATCH (b1:Book {id: $book1_id})
                            MATCH (b2:Book {id: $book2_id})
                            MERGE (b1)-[s:SIMILAR_TO]->(b2)
                            SET s.similarity_score = $similarity
                        """, 
                        book1_id=books_with_embeddings[i]['id'],
                        book2_id=books_with_embeddings[j]['id'],
                        similarity=float(similarity))
                        
                        relationships_created += 1
        
        print(f"Created {relationships_created} similarity relationships")
```

### Upload Statistics

```json
{
    "upload_summary": {
        "total_books": 2131,
        "books_uploaded": 2118,
        "books_failed": 13,
        "total_authors": 1456,
        "total_genres": 23,
        "total_publishers": 541,
        "relationships_created": {
            "wrote_relationships": 2118,
            "belongs_to_relationships": 4236,
            "published_relationships": 2011,
            "similarity_relationships": 4941
        },
        "upload_time_minutes": 1.5,
        "errors": [
            {
                "book_id": "book_123",
                "error": "Constraint violation: duplicate ISBN",
                "action": "skipped"
            }
        ],
        "warnings": [
            "Unknown property key: normalized_genres"
        ]
    }
}
```

## 📊 Pipeline Monitoring & Analytics

### Performance Metrics

```python
class PipelineMonitor:
    def __init__(self):
        self.metrics = {
            'collection': {},
            'cleaning': {},
            'merging': {},
            'embedding': {},
            'upload': {}
        }
    
    def track_stage_performance(self, stage, start_time, end_time, records_processed):
        """Tracks performance metrics for each pipeline stage"""
        
        duration = end_time - start_time
        throughput = records_processed / duration.total_seconds()
        
        self.metrics[stage] = {
            'duration_seconds': duration.total_seconds(),
            'records_processed': records_processed,
            'throughput_per_second': throughput,
            'timestamp': end_time.isoformat()
        }
    
    def generate_pipeline_report(self):
        """Generates comprehensive pipeline performance report"""
        
        total_duration = sum(stage['duration_seconds'] for stage in self.metrics.values())
        total_records = self.metrics.get('upload', {}).get('records_processed', 0)
        
        return {
            'pipeline_summary': {
                'total_duration_minutes': total_duration / 60,
                'total_records_processed': total_records,
                'overall_throughput': total_records / total_duration if total_duration > 0 else 0,
                'stage_breakdown': self.metrics
            }
        }
```

### Data Quality Dashboard

```python
def generate_quality_dashboard(books_data):
    """Generates data quality metrics dashboard"""
    
    quality_metrics = {
        'completeness': calculate_completeness(books_data),
        'consistency': calculate_consistency(books_data),
        'accuracy': calculate_accuracy(books_data),
        'uniqueness': calculate_uniqueness(books_data),
        'validity': calculate_validity(books_data)
    }
    
    return {
        'data_quality_score': sum(quality_metrics.values()) / len(quality_metrics),
        'metrics_breakdown': quality_metrics,
        'recommendations': generate_quality_recommendations(quality_metrics)
    }
```

## 🔧 Pipeline Configuration

### Configuration File Structure

```yaml
# pipeline_config.yaml
data_collection:
  max_pages_per_genre: 50
  rate_limit_delay: 2.0
  retry_attempts: 3
  timeout_seconds: 30
  
data_cleaning:
  min_title_length: 5
  max_description_length: 2000
  rating_scale: 5.0
  required_fields: ['title', 'author', 'genre']
  
deduplication:
  isbn_priority: true
  title_similarity_threshold: 0.9
  author_similarity_threshold: 0.8
  fuzzy_match_threshold: 85
  
embedding_generation:
  model: "text-embedding-ada-002"
  batch_size: 100
  max_tokens_per_text: 8000
  similarity_threshold: 0.7
  
database_upload:
  batch_size: 1000
  create_indexes: true
  create_similarity_relationships: true
  similarity_threshold: 0.7
```

### Environment-Specific Configurations

```python
# config.py
import os
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # Data Collection
    max_pages_per_genre: int = int(os.getenv('MAX_PAGES_PER_GENRE', 50))
    collection_delay: float = float(os.getenv('COLLECTION_DELAY', 2.0))
    
    # OpenAI
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    embedding_model: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
    embedding_batch_size: int = int(os.getenv('EMBEDDING_BATCH_SIZE', 100))
    
    # Neo4j
    neo4j_uri: str = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_username: str = os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password: str = os.getenv('NEO4J_PASSWORD', 'neo4j')
    
    # Quality Thresholds
    similarity_threshold: float = float(os.getenv('SIMILARITY_THRESHOLD', 0.7))
    min_rating_count: int = int(os.getenv('MIN_RATING_COUNT', 10))
```

## 🚀 Running the Complete Pipeline

### Automated Pipeline Execution

```python
# run_pipeline.py
import logging
from datetime import datetime
from pipeline_config import PipelineConfig

def run_complete_pipeline():
    """Executes the complete data pipeline"""
    
    config = PipelineConfig()
    monitor = PipelineMonitor()
    
    logging.info("Starting Book Recommendation System Data Pipeline")
    
    try:
        # Stage 1: Data Collection
        start_time = datetime.now()
        collected_books = run_data_collection(config)
        monitor.track_stage_performance('collection', start_time, datetime.now(), len(collected_books))
        
        # Stage 2: Data Cleaning
        start_time = datetime.now()
        cleaned_books = run_data_cleaning(collected_books, config)
        monitor.track_stage_performance('cleaning', start_time, datetime.now(), len(cleaned_books))
        
        # Stage 3: Data Merging
        start_time = datetime.now()
        merged_books = run_data_merging(cleaned_books, config)
        monitor.track_stage_performance('merging', start_time, datetime.now(), len(merged_books))
        
        # Stage 4: Embedding Generation
        start_time = datetime.now()
        books_with_embeddings = run_embedding_generation(merged_books, config)
        monitor.track_stage_performance('embedding', start_time, datetime.now(), len(books_with_embeddings))
        
        # Stage 5: Database Upload
        start_time = datetime.now()
        upload_results = run_database_upload(books_with_embeddings, config)
        monitor.track_stage_performance('upload', start_time, datetime.now(), upload_results['books_uploaded'])
        
        # Generate final report
        pipeline_report = monitor.generate_pipeline_report()
        save_pipeline_report(pipeline_report)
        
        logging.info("Pipeline completed successfully!")
        return pipeline_report
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_complete_pipeline()
```

### Command Line Interface

```bash
# Run complete pipeline
python run_pipeline.py

# Run specific stages
python run_pipeline.py --stage collection
python run_pipeline.py --stage embedding
python run_pipeline.py --stage upload

# Run with custom configuration
python run_pipeline.py --config custom_config.yaml

# Run in monitoring mode
python run_pipeline.py --monitor --output-dir ./reports
```

This comprehensive data pipeline documentation provides detailed insights into every aspect of the Book Recommendation System's data processing workflow, from initial collection through final database upload and analysis.