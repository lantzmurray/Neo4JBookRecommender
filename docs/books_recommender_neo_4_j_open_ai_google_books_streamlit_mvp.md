# Books Recommender — Neo4j + OpenAI + Google Books (Streamlit MVP)

Zorro-style blueprint to build a functional, compliant, hybrid recommender you can ship **today** with Streamlit, then upgrade to Next.js later. This doc doubles as TraeAI build spec.

---

## 0) Goal & Capabilities

**Goal:** Ship a working app that recommends books using:
- **Backbone**: Your Kaggle Kindle dataset (offline, ~130k) ingested into Neo4j
- **Freshness**: Google Books API (no scraping) for **2024+** new releases
- **Search/RAG-style retrieval**: OpenAI embeddings stored in Neo4j vector index (or GDS kNN fallback)
- **UI**: Streamlit tabs — *Explore*, *Fresh*, *My Recs*

**Key constraints:**
- Single Neo4j **AuraDB Free** instance; multiple apps co-resident
- No HTML scraping; compliant APIs only
- Cost-aware (use `text-embedding-3-small`)

---

## 1) Tech Stack

- **Neo4j** 5.x (AuraDB Free), Neo4j Python Driver
- **OpenAI** embeddings: `text-embedding-3-small` (1536-d)
- **Google Books API** (public endpoint; optional API key if you have one)
- **Streamlit** for UI
- **Python** 3.10–3.12

---

## 2) Repository Layout (TraeAI: create exactly this)

```
books-recs/
├─ app/
│  └─ streamlit_app.py
├─ ingest/
│  ├─ ingest_kaggle_books.py
│  └─ google_books_ingest.py
├─ neo4j/
│  └─ cypher/
│     ├─ constraints.cypher
│     ├─ vector_index.cypher
│     └─ tag_xerox.cypher
├─ docs/
│  ├─ README.md
│  ├─ SETUP.md
│  ├─ DATA_MODEL.md
│  ├─ KAGGLE_INGEST.md
│  ├─ GOOGLE_BOOKS.md
│  ├─ EMBEDDINGS.md
│  ├─ VECTOR_SEARCH.md
│  ├─ STREAMLIT_UI.md
│  ├─ EVALUATION.md
│  └─ ROADMAP.md
├─ config/
│  └─ .env.example
├─ requirements.txt
└─ Makefile
```

> This single document includes the full text for each file below. Copy as-is into your repo.

---

## 3) Files & Content

### 3.1 `requirements.txt`
```
streamlit==1.38.0
neo4j==5.23.0
openai==1.42.0
pandas==2.2.2
python-dotenv==1.0.1
isbnlib==3.10.14
requests==2.32.3
```

### 3.2 `config/.env.example`
```
# Neo4j Aura
NEO4J_URI=neo4j+s://<your-uri>
NEO4J_USER=neo4j
NEO4J_PASS=<password>

# OpenAI
OPENAI_API_KEY=sk-<...>

# Google Books (optional; not required for public endpoint)
GOOGLE_BOOKS_API_KEY=

# Categories for Google Books nightly ingest (comma-separated)
GBOOKS_CATEGORIES=Horror,Thriller,Science Fiction
GBOOKS_MIN_YEAR=2024
GBOOKS_PER_CATEGORY=60

# Kaggle CSV local path (adjust when running ingest)
KAGGLE_CSV=./amazon-kindle-books-2023.csv
```

### 3.3 `neo4j/cypher/constraints.cypher`
```cypher
// Namespace index (used by Xerox + Books separation)
CREATE INDEX IF NOT EXISTS FOR (n) ON (n.app);

// Unique IDs
CREATE CONSTRAINT books_book_id IF NOT EXISTS
FOR (b:Books_Book) REQUIRE (b.app, b.bookId) IS UNIQUE;

CREATE CONSTRAINT books_author_id IF NOT EXISTS
FOR (a:Books_Author) REQUIRE (a.app, a.authorId) IS UNIQUE;

CREATE CONSTRAINT books_user_id IF NOT EXISTS
FOR (u:Books_User) REQUIRE (u.app, u.userId) IS UNIQUE;
```

### 3.4 `neo4j/cypher/vector_index.cypher`
```cypher
// Neo4j native vector index (1536-d for text-embedding-3-small)
CREATE VECTOR INDEX books_embedding IF NOT EXISTS
FOR (b:Books_Book) ON (b.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};
```

### 3.5 `neo4j/cypher/tag_xerox.cypher` (one-time safety)
```cypher
MATCH (n) SET n.app = coalesce(n.app, 'xerox');
MATCH ()-[r]->() SET r.app = coalesce(r.app, 'xerox');
```

---

### 3.6 `docs/DATA_MODEL.md`

**Nodes**
- `(:Books_Book {app:'books', bookId, title, desc, genres:[], year, publishedDate, pages, ratingAvg, ratingCount, coverUrl, publisher, source, embedding:[...]})`
- `(:Books_Author {app:'books', authorId, name})`
- `(:Books_User {app:'books', userId, name})`

**Relationships**
- `(:Books_Author)-[:BOOKS_WROTE {app:'books'}]->(:Books_Book)`
- `(:Books_User)-[:BOOKS_RATED {app:'books', stars, ts, shelves:[], source}]->(:Books_Book)`
- Optional similarity edges if using GDS fallback: `(:Books_Book)-[:BOOKS_SIMILAR {score}]→(:Books_Book)`

**IDs**
- Prefer `isbn13`; fallback to `asin`; else hash(`title|authors`).

**Separation**
- Everything under **books** has `app:'books'` + labels `Books_*`. Xerox data stays `app:'xerox'`.

---

### 3.7 `docs/SETUP.md`

**Prereqs**
- Neo4j AuraDB Free instance (get `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`)
- OpenAI API key
- Python 3.10+ and `virtualenv`

**Steps**
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
2. `pip install -r requirements.txt`
3. Copy `config/.env.example` → `.env` and fill values
4. Run constraints:
   - Open Neo4j Browser → paste `neo4j/cypher/constraints.cypher`
   - (Optional) tag existing nodes as Xerox: run `neo4j/cypher/tag_xerox.cypher`
5. Create vector index: run `neo4j/cypher/vector_index.cypher`

---

### 3.8 `ingest/ingest_kaggle_books.py`
```python
import os, hashlib
from typing import Dict, Any, List
import pandas as pd
from isbnlib import to_isbn13
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
CSV_PATH   = os.getenv("KAGGLE_CSV", "./amazon-kindle-books-2023.csv")

# Map after inspecting your CSV headers (adjust as needed)
FIELD_MAP = {
    "title": "title",
    "authors": "author",
    "asin": "asin",
    "isbn10": "isbn10",
    "isbn13": "isbn13",
    "description": "description",
    "categories": "categories",
    "ratingAvg": "rating",
    "ratingCount": "reviews_count",
    "pages": "pages",
    "publisher": "publisher",
    "publishedYear": "year"
}

UPSERT = """
UNWIND $rows AS r
MERGE (b:Books_Book {app:'books', bookId:r.bookId})
SET b += {
  asin:r.asin, isbn10:r.isbn10, isbn13:r.isbn13,
  title:r.title, desc:r.desc, genres:r.genres,
  year:r.year, pages:r.pages, ratingAvg:r.ratingAvg,
  ratingCount:r.ratingCount, coverUrl:r.coverUrl,
  publisher:r.publisher, source:'kaggle'
}
WITH b, r
UNWIND r.authors AS aName
  WITH b, apoc.text.trim(aName) AS aName
  WHERE aName IS NOT NULL AND aName <> ''
  MERGE (a:Books_Author {app:'books', authorId: apoc.text.clean(aName)})
  SET a.name = aName
  MERGE (a)-[:BOOKS_WROTE {app:'books'}]->(b);
"""

def stable_book_id(row: Dict[str, Any]) -> str:
    for key in ("isbn13","asin"):
        v = row.get(key)
        if v: return v
    basis = f"{row.get('title','')}|{'&'.join(row.get('authors',[]))}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

def normalize_isbn13(isbn10: str|None, isbn13: str|None) -> str|None:
    if isbn13: return isbn13.replace("-","")
    if isbn10:
        try: return to_isbn13(isbn10)
        except: return None
    return None

def parse_authors(val) -> List[str]:
    if val is None: return []
    s = str(val)
    for sep in [";","|"]:
        s = s.replace(sep, ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts

def normalize_genres(val) -> List[str]:
    if val is None: return []
    s = str(val)
    for sep in [";","|"]:
        s = s.replace(sep, ",")
    return list({g.strip() for g in s.split(",") if g.strip()})

def main():
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns={v:k for k,v in FIELD_MAP.items() if v in df.columns})

    rows = []
    for _, r in df.iterrows():
        title = str(r.get("title") or "").strip()
        if not title: continue
        authors = parse_authors(r.get("authors"))
        genres  = normalize_genres(r.get("categories") if "categories" in r else r.get("genres"))
        isbn13  = normalize_isbn13(r.get("isbn10"), r.get("isbn13"))
        asin    = r.get("asin") or None
        book = {
            "asin": asin,
            "isbn10": r.get("isbn10") or None,
            "isbn13": isbn13,
            "title": title[:512],
            "desc": (r.get("description") or r.get("summary") or None),
            "genres": genres,
            "year": int(r.get("publishedYear")) if pd.notna(r.get("publishedYear")) else None,
            "pages": int(float(r.get("pages"))) if pd.notna(r.get("pages")) else None,
            "ratingAvg": float(r.get("ratingAvg")) if pd.notna(r.get("ratingAvg")) else None,
            "ratingCount": int(float(r.get("ratingCount"))) if pd.notna(r.get("ratingCount")) else None,
            "coverUrl": r.get("cover") if "cover" in r else None,
            "publisher": r.get("publisher") or None,
            "authors": authors
        }
        book["bookId"] = stable_book_id(book)
        rows.append(book)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        s.run(UPSERT, rows=rows)
    driver.close()
    print(f"Ingested {len(rows)} Kaggle books into app:'books'.")

if __name__ == "__main__":
    main()
```

### 3.9 `ingest/google_books_ingest.py`
```python
import os, time, hashlib, requests
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GBOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")
CATEGORIES = [c.strip() for c in os.getenv("GBOOKS_CATEGORIES","Horror,Thriller,Science Fiction").split(',') if c.strip()]
MIN_YEAR = int(os.getenv("GBOOKS_MIN_YEAR","2024"))
PER_CAT = int(os.getenv("GBOOKS_PER_CATEGORY","60"))

UPSERT = """
UNWIND $rows AS r
MERGE (b:Books_Book {app:'books', bookId:r.bookId})
SET b += {
  title:r.title, desc:r.desc, genres:r.genres, year:r.year,
  publishedDate:r.publishedDate, coverUrl:r.coverUrl, publisher:r.publisher,
  source:'google-books', embedding:r.embedding
}
WITH b, r
UNWIND r.authors AS aName
  MERGE (a:Books_Author {app:'books', authorId: apoc.text.clean(aName)})
  SET a.name = aName
  MERGE (a)-[:BOOKS_WROTE {app:'books'}]->(b);
"""

def embed(text: str) -> List[float]:
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return resp.data[0].embedding

def parse_year(pd: str|None) -> int|None:
    if not pd: return None
    try: return int(pd[:4])
    except: return None

def normalize_item(item: Dict[str,Any]) -> Dict[str,Any]:
    info = item.get("volumeInfo", {})
    title = info.get("title")
    desc  = info.get("description")
    authors = info.get("authors") or []
    cats    = info.get("categories") or []
    pd      = info.get("publishedDate")
    year    = parse_year(pd)
    cover   = (info.get("imageLinks") or {}).get("thumbnail")
    pub     = info.get("publisher")
    bid = None
    for ident in info.get("industryIdentifiers", []) or []:
        if ident.get("type") in ("ISBN_13","ISBN_10"):
            bid = ident.get("identifier"); break
    if not bid:
        bid = item.get("id") or hashlib.sha1(f"{title}|{','.join(authors)}".encode()).hexdigest()[:16]
    return {
        "bookId": bid, "title": title, "desc": desc, "authors": authors,
        "genres": cats, "publishedDate": pd, "year": year, "coverUrl": cover, "publisher": pub
    }

def fetch_newest(cat: str, limit: int) -> List[Dict[str,Any]]:
    out, start = [], 0
    while len(out) < limit and start < 200:
        params = {"q": f"subject:{cat}", "orderBy": "newest", "maxResults": 40, "startIndex": start, "printType": "books"}
        if GBOOKS_API_KEY: params["key"] = GBOOKS_API_KEY
        j = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=15).json()
        items = j.get("items", [])
        if not items: break
        for it in items:
            row = normalize_item(it)
            if row["year"] and row["year"] >= MIN_YEAR:
                out.append(row)
                if len(out) >= limit: break
        start += 40
        time.sleep(0.2)
    return out

def main():
    all_rows = []
    for cat in CATEGORIES:
        rows = fetch_newest(cat, PER_CAT)
        for r in rows:
            text = (r["title"] or "") + " " + (r.get("desc") or "")
            r["embedding"] = embed(text)
        all_rows.extend(rows)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    with driver.session() as s:
        s.run(UPSERT, rows=all_rows)
    driver.close()
    print(f"Upserted {len(all_rows)} Google Books rows (year >= {MIN_YEAR}).")

if __name__ == "__main__":
    main()
```

### 3.10 `docs/EMBEDDINGS.md`

**Batch populate embeddings for Kaggle books** (only if you want immediate semantic search before Google Books ingest):

```python
# tools/batch_embed_kaggle.py (optional helper)
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

BATCH = 200

def embed(text: str):
    import openai
    openai.api_key = OPENAI_API_KEY
    r = openai.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return r.data[0].embedding

with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS)) as driver:
    with driver.session() as s:
        while True:
            rows = s.run("""
                MATCH (b:Books_Book {app:'books'})
                WHERE b.embedding IS NULL
                WITH b LIMIT $lim
                RETURN b.bookId AS id, coalesce(b.title,'') + ' ' + coalesce(b.desc,'') AS text
            """, lim=BATCH).data()
            if not rows: break
            updates = []
            for row in rows:
                vec = embed(row['text'])
                updates.append({"bookId": row['id'], "embedding": vec})
            s.run("""
                UNWIND $rows AS r
                MATCH (b:Books_Book {app:'books', bookId:r.bookId})
                SET b.embedding = r.embedding
            """, rows=updates)
            print(f"Embedded {len(updates)}")
```

Notes:
- Use `text-embedding-3-small` for low cost; cache results and only embed once per book.
- Ensure `books_embedding` vector index exists before vector queries.

---

### 3.12 `docs/VECTOR_SEARCH.md`

**Local vector search (Neo4j native)**
```cypher
WITH $qEmb AS qv
CALL db.index.vector.queryNodes('books_embedding', 25, qv)
YIELD node, score
WHERE node.app = 'books'
RETURN node.title AS title, node.year AS year, node.genres AS genres, score
ORDER BY score DESC LIMIT 10;
```

**GDS fallback (if vector index not available)**
```cypher
CALL gds.graph.project('booksGraph', {Books_Book:{properties:['embedding']}}, {});
CALL gds.knn.write('booksGraph', {
  nodeProperties:['embedding'], topK:25, similarityFunction:'COSINE',
  writeRelationshipType:'BOOKS_SIMILAR', writeProperty:'score'
});
// Query
MATCH (seed:Books_Book {app:'books', bookId:$seed})- [s:BOOKS_SIMILAR]->(rec:Books_Book)
RETURN rec.title, s.score ORDER BY s.score DESC LIMIT 10;
```

---

### 3.13 `app/streamlit_app.py`
```python
import os, math, requests, tempfile
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GBOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

@st.cache_resource
def neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def embed(text: str):
    import openai
    openai.api_key = OPENAI_API_KEY
    r = openai.embeddings.create(model="text-embedding-3-small", input=text[:8000])
    return r.data[0].embedding

def cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    num = sum(x*y for x,y in zip(a,b))
    da = math.sqrt(sum(x*x for x in a)); db = math.sqrt(sum(y*y for y in b))
    return num/(da*db+1e-9)

LOCAL_QUERY = """
WITH $qEmb AS qv
CALL db.index.vector.queryNodes('books_embedding', $k, qv)
YIELD node, score
WHERE node.app = 'books'
RETURN node.title AS title, node.year AS year, node.genres AS genres, node.coverUrl AS coverUrl,
       score AS vecScore, node.source AS source
ORDER BY vecScore DESC
"""

def search_local(query: str, k: int=25):
    qv = embed(query)
    with neo4j_driver().session() as s:
        rows = s.run(LOCAL_QUERY, qEmb=qv, k=k).data()
    return rows

def google_books_newest(q: str, min_year=2024):
    params = {"q": q, "orderBy": "newest", "maxResults": 40, "printType": "books"}
    if GBOOKS_API_KEY: params["key"] = GBOOKS_API_KEY
    items = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=15).json().get("items", [])
    out = []
    for it in items:
        info = it.get("volumeInfo", {})
        title = info.get("title"); desc = info.get("description")
        year = None
        try:
            if info.get("publishedDate"): year = int(info["publishedDate"][:4])
        except: pass
        if not year or year < min_year: continue
        cover = (info.get("imageLinks") or {}).get("thumbnail")
        emb = embed((title or "") + " " + (desc or ""))
        out.append({
            "title": title, "year": year, "genres": info.get("categories", []),
            "coverUrl": cover, "source": "google-books", "embedding": emb
        })
    return out

st.set_page_config(page_title="Books Recommender", layout="wide")
st.title("📚 Books Recommender — Neo4j + OpenAI + Google Books")

TAB_EXPLORE, TAB_FRESH, TAB_MYRECS = st.tabs(["Explore", "Fresh (2024+)", "My Recs"])

with TAB_EXPLORE:
    q = st.text_input("Describe what you want (themes, vibes, comps)", "fast-paced horror with unreliable narrator")
    k = st.slider("Results", 5, 50, 15)
    if st.button("Search local"):    
        rows = search_local(q, k)
        for r in rows:
            cols = st.columns([1,4])
            with cols[0]:
                if r.get("coverUrl"): st.image(r["coverUrl"], use_container_width=True)
            with cols[1]:
                st.markdown(f"**{r['title']}** ({r.get('year','—')})  ")
                st.caption(", ".join(r.get("genres") or []))
                st.write(f"Source: {r.get('source','neo4j')}  |  Score: {r['vecScore']:.2f}")

with TAB_FRESH:
    qf = st.text_input("What fresh 2024+ are you in the mood for?", "horror motherhood unreliable narrator")
    if st.button("Find newest"):
        rows = google_books_newest(qf, min_year=2024)
        qv = embed(qf)
        # Rerank by cosine to query
        rows.sort(key=lambda x: cosine(qv, x["embedding"]), reverse=True)
        for r in rows[:20]:
            cols = st.columns([1,4])
            with cols[0]:
                if r.get("coverUrl"): st.image(r["coverUrl"], use_container_width=True)
            with cols[1]:
                st.markdown(f"**{r['title']}** ({r.get('year','—')})  ")
                st.caption(", ".join(r.get("genres") or []))
                st.write("Source: Google Books (2024+)")

with TAB_MYRECS:
    st.info("Personalized recommendations based on your reading history and preferences.")
```

---

### 3.14 `docs/GOOGLE_BOOKS.md`

- We use **public Google Books API** with `orderBy=newest`, filter client-side for `year >= MIN_YEAR`.
- No scraping; optional API key increases quota.
- Nightly job: `python ingest/google_books_ingest.py` (CRON or GitHub Action).

---

### 3.15 `docs/STREAMLIT_UI.md`

Tabs:
- **Explore**: vector search over local graph (Kaggle + Google Books cached)
- **Fresh (2024+)**: live Google Books newest → embed → rerank
- **My Recs**: personalized recommendations based on reading history and preferences

Design later: migrate to Next.js; preserve API surface.

---

### 3.17 `docs/EVALUATION.md`

- **Click-through rate** on top-5 suggestions
- **Save-to-shelf rate**
- **Diversity**: distinct authors/genres in top-10
- **Freshness coverage**: share of results with `year >= 2024`

---

### 3.18 `docs/ROADMAP.md`

**V1 → V1.5**
- Add taste vector computation & hybrid rerank (vector + authorBoost + shelf match)
- Cache fresh Google Books hits back to Neo4j with `lastChecked`

**V2**
- Migrate UI to Next.js (same endpoints), add server-side streaming
- Add Keepa or Hardcover as second fresh source
- Theme/trope KG via Neo4j LLM KG Builder (on allowed texts)

---

### 3.19 `Makefile`
```
.PHONY: setup kaggle google fresh app
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

kaggle:
	python ingest/ingest_kaggle_books.py

google:
	python ingest/google_books_ingest.py

app:
	streamlit run app/streamlit_app.py
```

---

## 4) Step-by-Step Checklist (copy into TraeAI runbook)

- [ ] Create AuraDB Free, get URI/user/pass
- [ ] Set `.env`
- [ ] Run `constraints.cypher` & `vector_index.cypher`
- [ ] (Optional) `tag_xerox.cypher`
- [ ] Download Kaggle CSV → set `KAGGLE_CSV` → `make kaggle`
- [ ] (Optional) `tools/batch_embed_kaggle.py` to embed backlog
- [ ] `make google` to add newest 2024+ per category
- [ ] `make app` → test **Explore** & **Fresh** tabs
- [ ] Import Goodreads CSV in **Import** tab
- [ ] Add taste-vector logic (V1.5) and a *My Recs* query

---

## 5) Guardrails & Notes
- Respect API quotas; cache embeddings; only write vectors once per book
- Keep `app:'books'` on every node/rel to avoid mixing with Xerox
- Prefer ISBN13 as `bookId` when available
- For Aura Free, avoid heavy batch sizes; throttle requests

---

## 6) Accountability (light + humane)
- **MIT today (≤45 min):** run constraints, ingest Kaggle, launch Streamlit
- **Proverbs 21:5** — steady planning ➜ abundance
- **Turn toward**: ask Angela to pick 5 horror queries to test in *Fresh* tab tonight

