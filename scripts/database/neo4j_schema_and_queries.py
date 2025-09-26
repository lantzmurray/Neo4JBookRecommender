#!/usr/bin/env python3
"""
Neo4j Schema Design and Cypher Queries for Book Recommendation System
"""

# Neo4j Schema Design
NEO4J_SCHEMA = {
    "nodes": {
        "Book": {
            "properties": [
                "id",           # Unique identifier
                "title",        # Book title
                "isbn",         # ISBN-10
                "isbn13",       # ISBN-13
                "year_published",
                "pages",
                "average_rating",
                "ratings_count",
                "description",
                "publisher",
                "google_id",
                "goodreads_id",
                "popularity_score",
                "embedding",    # Vector embedding for similarity
                "tag"          # 'bookrecommender' to separate from xerox docs
            ],
            "indexes": [
                "CREATE INDEX book_title_idx FOR (b:Book) ON (b.title)",
                "CREATE INDEX book_tag_idx FOR (b:Book) ON (b.tag)",
                "CREATE INDEX book_rating_idx FOR (b:Book) ON (b.average_rating)",
                "CREATE INDEX book_year_idx FOR (b:Book) ON (b.year_published)"
            ]
        },
        "Author": {
            "properties": [
                "id",
                "name",
                "tag"
            ],
            "indexes": [
                "CREATE INDEX author_name_idx FOR (a:Author) ON (a.name)",
                "CREATE INDEX author_tag_idx FOR (a:Author) ON (a.tag)"
            ]
        },
        "Genre": {
            "properties": [
                "id",
                "name",
                "tag"
            ],
            "indexes": [
                "CREATE INDEX genre_name_idx FOR (g:Genre) ON (g.name)",
                "CREATE INDEX genre_tag_idx FOR (g:Genre) ON (g.tag)"
            ]
        },
        "Publisher": {
            "properties": [
                "id",
                "name",
                "tag"
            ],
            "indexes": [
                "CREATE INDEX publisher_name_idx FOR (p:Publisher) ON (p.name)",
                "CREATE INDEX publisher_tag_idx FOR (p:Publisher) ON (p.tag)"
            ]
        },
        "User": {
            "properties": [
                "id",
                "name",
                "tag"
            ],
            "indexes": [
                "CREATE INDEX user_id_idx FOR (u:User) ON (u.id)",
                "CREATE INDEX user_tag_idx FOR (u:User) ON (u.tag)"
            ]
        }
    },
    "relationships": {
        "WROTE": {
            "from": "Author",
            "to": "Book",
            "properties": []
        },
        "BELONGS_TO": {
            "from": "Book", 
            "to": "Genre",
            "properties": []
        },
        "PUBLISHED_BY": {
            "from": "Book",
            "to": "Publisher",
            "properties": []
        },
        "READ": {
            "from": "User",
            "to": "Book",
            "properties": [
                "rating",
                "read_date",
                "shelves"
            ]
        },
        "SIMILAR_TO": {
            "from": "Book",
            "to": "Book",
            "properties": [
                "similarity_score",
                "similarity_type"  # 'content', 'collaborative', 'hybrid'
            ]
        }
    }
}

# Cypher Queries for Book Recommendation System
CYPHER_QUERIES = {
    
    # Data Creation Queries
    "create_book": """
        CREATE (b:Book {
            id: $id,
            title: $title,
            isbn: $isbn,
            isbn13: $isbn13,
            year_published: $year_published,
            pages: $pages,
            average_rating: $average_rating,
            ratings_count: $ratings_count,
            description: $description,
            publisher: $publisher,
            google_id: $google_id,
            goodreads_id: $goodreads_id,
            popularity_score: $popularity_score,
            embedding: $embedding,
            tag: 'bookrecommender'
        })
        RETURN b
    """,
    
    "create_author": """
        MERGE (a:Author {name: $name, tag: 'bookrecommender'})
        RETURN a
    """,
    
    "create_genre": """
        MERGE (g:Genre {name: $name, tag: 'bookrecommender'})
        RETURN g
    """,
    
    "create_publisher": """
        MERGE (p:Publisher {name: $name, tag: 'bookrecommender'})
        RETURN p
    """,
    
    "create_relationships": """
        MATCH (b:Book {id: $book_id, tag: 'bookrecommender'})
        MATCH (a:Author {name: $author_name, tag: 'bookrecommender'})
        MERGE (a)-[:WROTE]->(b)
    """,
    
    "create_genre_relationship": """
        MATCH (b:Book {id: $book_id, tag: 'bookrecommender'})
        MATCH (g:Genre {name: $genre_name, tag: 'bookrecommender'})
        MERGE (b)-[:BELONGS_TO]->(g)
    """,
    
    "create_publisher_relationship": """
        MATCH (b:Book {id: $book_id, tag: 'bookrecommender'})
        MATCH (p:Publisher {name: $publisher_name, tag: 'bookrecommender'})
        MERGE (b)-[:PUBLISHED_BY]->(p)
    """,
    
    # Recommendation Queries
    "content_based_recommendations": """
        MATCH (target:Book {id: $book_id, tag: 'bookrecommender'})
        MATCH (similar:Book {tag: 'bookrecommender'})
        WHERE target <> similar
        WITH target, similar, 
             gds.similarity.cosine(target.embedding, similar.embedding) AS similarity
        WHERE similarity > $threshold
        RETURN similar.title AS title, 
               similar.author AS author,
               similar.average_rating AS rating,
               similarity
        ORDER BY similarity DESC
        LIMIT $limit
    """,
    
    "genre_based_recommendations": """
        MATCH (target:Book {id: $book_id, tag: 'bookrecommender'})-[:BELONGS_TO]->(g:Genre)
        MATCH (similar:Book {tag: 'bookrecommender'})-[:BELONGS_TO]->(g)
        WHERE target <> similar
        RETURN similar.title AS title,
               similar.author AS author, 
               similar.average_rating AS rating,
               similar.popularity_score AS popularity,
               collect(g.name) AS shared_genres
        ORDER BY similar.popularity_score DESC, similar.average_rating DESC
        LIMIT $limit
    """,
    
    "author_based_recommendations": """
        MATCH (target:Book {id: $book_id, tag: 'bookrecommender'})<-[:WROTE]-(a:Author)
        MATCH (a)-[:WROTE]->(similar:Book {tag: 'bookrecommender'})
        WHERE target <> similar
        RETURN similar.title AS title,
               similar.author AS author,
               similar.average_rating AS rating,
               similar.popularity_score AS popularity
        ORDER BY similar.popularity_score DESC, similar.average_rating DESC
        LIMIT $limit
    """,
    
    "hybrid_recommendations": """
        MATCH (target:Book {id: $book_id, tag: 'bookrecommender'})
        MATCH (similar:Book {tag: 'bookrecommender'})
        WHERE target <> similar
        
        // Content similarity
        WITH target, similar,
             gds.similarity.cosine(target.embedding, similar.embedding) AS content_sim
        
        // Genre overlap
        OPTIONAL MATCH (target)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(similar)
        WITH target, similar, content_sim, count(g) AS genre_overlap
        
        // Author match
        OPTIONAL MATCH (target)<-[:WROTE]-(a:Author)-[:WROTE]->(similar)
        WITH target, similar, content_sim, genre_overlap, count(a) AS author_match
        
        // Calculate hybrid score
        WITH similar,
             (content_sim * 0.4 + 
              (genre_overlap * 0.1) + 
              (author_match * 0.2) + 
              (similar.popularity_score / 10.0 * 0.2) +
              (similar.average_rating / 5.0 * 0.1)) AS hybrid_score
        
        WHERE hybrid_score > $threshold
        RETURN similar.title AS title,
               similar.author AS author,
               similar.average_rating AS rating,
               similar.popularity_score AS popularity,
               hybrid_score
        ORDER BY hybrid_score DESC
        LIMIT $limit
    """,
    
    # Discovery Queries
    "trending_books": """
        MATCH (b:Book {tag: 'bookrecommender'})
        WHERE b.year_published >= $year_threshold
        RETURN b.title AS title,
               b.author AS author,
               b.average_rating AS rating,
               b.ratings_count AS ratings_count,
               b.popularity_score AS popularity
        ORDER BY b.popularity_score DESC, b.average_rating DESC
        LIMIT $limit
    """,
    
    "top_rated_by_genre": """
        MATCH (b:Book {tag: 'bookrecommender'})-[:BELONGS_TO]->(g:Genre {name: $genre})
        WHERE b.ratings_count >= $min_ratings
        RETURN b.title AS title,
               b.author AS author,
               b.average_rating AS rating,
               b.ratings_count AS ratings_count,
               b.year_published AS year
        ORDER BY b.average_rating DESC, b.ratings_count DESC
        LIMIT $limit
    """,
    
    "books_by_author": """
        MATCH (a:Author {name: $author_name, tag: 'bookrecommender'})-[:WROTE]->(b:Book)
        RETURN b.title AS title,
               b.average_rating AS rating,
               b.year_published AS year,
               b.popularity_score AS popularity
        ORDER BY b.year_published DESC, b.popularity_score DESC
    """,
    
    # User Interaction Queries
    "user_recommendations_based_on_history": """
        MATCH (u:User {id: $user_id, tag: 'bookrecommender'})-[r:READ]->(read_book:Book)
        WHERE r.rating >= $min_user_rating
        
        // Find similar books to highly rated ones
        MATCH (read_book)-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(rec:Book {tag: 'bookrecommender'})
        WHERE NOT EXISTS((u)-[:READ]->(rec))
        
        WITH rec, avg(read_book.average_rating) AS avg_user_pref, count(*) AS genre_matches
        WHERE rec.average_rating >= avg_user_pref - 0.5
        
        RETURN rec.title AS title,
               rec.author AS author,
               rec.average_rating AS rating,
               rec.popularity_score AS popularity,
               genre_matches
        ORDER BY genre_matches DESC, rec.popularity_score DESC
        LIMIT $limit
    """,
    
    # Analytics Queries
    "genre_statistics": """
        MATCH (b:Book {tag: 'bookrecommender'})-[:BELONGS_TO]->(g:Genre)
        RETURN g.name AS genre,
               count(b) AS book_count,
               avg(b.average_rating) AS avg_rating,
               avg(b.popularity_score) AS avg_popularity
        ORDER BY book_count DESC
    """,
    
    "author_statistics": """
        MATCH (a:Author {tag: 'bookrecommender'})-[:WROTE]->(b:Book)
        RETURN a.name AS author,
               count(b) AS book_count,
               avg(b.average_rating) AS avg_rating,
               max(b.popularity_score) AS max_popularity
        ORDER BY book_count DESC
        LIMIT $limit
    """,
    
    # Verification Queries
    "check_xerox_docs": """
        MATCH (n)
        WHERE NOT n.tag = 'bookrecommender'
        RETURN labels(n) AS node_type, count(n) AS count
    """,
    
    "count_book_nodes": """
        MATCH (b:Book {tag: 'bookrecommender'})
        RETURN count(b) AS total_books
    """,
    
    # Cleanup Queries
    "delete_book_recommender_data": """
        MATCH (n {tag: 'bookrecommender'})
        DETACH DELETE n
    """,
    
    "create_similarity_relationships": """
        MATCH (b1:Book {tag: 'bookrecommender'})
        MATCH (b2:Book {tag: 'bookrecommender'})
        WHERE id(b1) < id(b2)
        WITH b1, b2, gds.similarity.cosine(b1.embedding, b2.embedding) AS similarity
        WHERE similarity > $threshold
        CREATE (b1)-[:SIMILAR_TO {similarity_score: similarity, similarity_type: 'content'}]->(b2)
        CREATE (b2)-[:SIMILAR_TO {similarity_score: similarity, similarity_type: 'content'}]->(b1)
    """
}

# Vector Search Queries (for Neo4j with vector indexes)
VECTOR_QUERIES = {
    "create_vector_index": """
        CREATE VECTOR INDEX book_embeddings 
        FOR (b:Book) ON (b.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
    """,
    
    "vector_similarity_search": """
        CALL db.index.vector.queryNodes('book_embeddings', $k, $query_embedding)
        YIELD node AS similar_book, score
        WHERE similar_book.tag = 'bookrecommender'
        RETURN similar_book.title AS title,
               similar_book.author AS author,
               similar_book.average_rating AS rating,
               score
        ORDER BY score DESC
    """
}

def print_schema_summary():
    """Print a summary of the Neo4j schema"""
    print("=== Neo4j Schema for Book Recommendation System ===\n")
    
    print("NODES:")
    for node_type, details in NEO4J_SCHEMA["nodes"].items():
        print(f"  {node_type}:")
        print(f"    Properties: {', '.join(details['properties'])}")
        print()
    
    print("RELATIONSHIPS:")
    for rel_type, details in NEO4J_SCHEMA["relationships"].items():
        print(f"  {details['from']} -[:{rel_type}]-> {details['to']}")
        if details['properties']:
            print(f"    Properties: {', '.join(details['properties'])}")
        print()
    
    print(f"TOTAL QUERIES DEFINED: {len(CYPHER_QUERIES)}")
    print(f"VECTOR QUERIES: {len(VECTOR_QUERIES)}")

if __name__ == "__main__":
    print_schema_summary()
    
    print("\nKey Features:")
    print("- Separate 'bookrecommender' tag to avoid conflicts with xerox docs")
    print("- Vector embeddings for content-based recommendations")
    print("- Hybrid recommendation scoring")
    print("- User interaction tracking")
    print("- Comprehensive analytics queries")
    print("- Genre, author, and publisher relationships")
    print("- Similarity relationships between books")