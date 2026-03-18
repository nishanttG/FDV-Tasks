#  TactIQ Day 2: Complete Visual Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🚀 DAY 2 PIPELINE OVERVIEW                       │
└─────────────────────────────────────────────────────────────────────┘

                         ╔════════════════════╗
                         ║   INPUT DATA       ║
                         ╚════════════════════╝
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │ Player Stats │      │ Blog Articles│      │ Team Stats   │
  │  19,388 rows │      │  17 articles │      │  (Future)    │
  └──────────────┘      └──────────────┘      └──────────────┘
          │                       │
          │                       │
          ▼                       ▼
  ┌──────────────────────────────────────────┐
  │   TABLE-TO-TEXT CONVERSION               │
  │   (Natural Language Descriptions)        │
  └──────────────────────────────────────────┘
          │                       │
          └───────────┬───────────┘
                      ▼
          ╔═══════════════════════════╗
          ║   EMBEDDING PIPELINE      ║
          ║   all-MiniLM-L6-v2        ║
          ║   384 dimensions          ║
          ╚═══════════════════════════╝
                      │
                      ▼
          ┌───────────────────────────┐
          │   BATCH PROCESSING        │
          │   500 docs/batch          │
          │   ~70 docs/s CPU          │
          └───────────────────────────┘
                      │
                      ▼
          ╔═══════════════════════════╗
          ║   CHROMADB STORAGE        ║
          ║   Persistent Local        ║
          ║   db/chroma/              ║
          ╚═══════════════════════════╝
                      │
                      ▼
          ┌───────────────────────────┐
          │   SEMANTIC SEARCH         │
          │   Query → Top-K Results   │
          │   Similarity Scoring      │
          └───────────────────────────┘
                      │
                      ▼
          ╔═══════════════════════════╗
          ║   OUTPUT / VALIDATION     ║
          ║   19,405 docs indexed     ║
          ║   Test queries validated  ║
          ╚═══════════════════════════╝
```

---

##  Data Flow Diagram

```
INPUT FILES                      PROCESSING                    OUTPUT
─────────────                   ──────────────               ───────────

player_stats_                   create_player_               ChromaDB:
unified_CLEANED.csv   ──────►   description()    ──────►     - player_123
(19,388 rows)                   "Haaland is a..."            - player_456
                                                             - player_789
                                                              ...
                                     ▲
                                     │
tactical_blogs_                 Full article      ──────►     - blog_1
20251229.json        ──────►    text ingestion               - blog_2
(17 articles)                   (no conversion)              - blog_3
                                                              ...

                                     │
                                     ▼
                            ┌──────────────────┐
                            │ Embedding Model  │
                            │ all-MiniLM-L6-v2 │
                            │ 384-dim vectors  │
                            └──────────────────┘
```

---

##  Execution Flow

### Script Approach:
```
Terminal Command
       │
       ▼
┌──────────────────────────────────────────────┐
│  python script/day2_embed_and_ingest.py      │
└──────────────────────────────────────────────┘
       │
       ├─► Initialize EmbeddingPipeline
       ├─► Initialize VectorDatabase
       ├─► Load player CSV (19,388 rows)
       ├─► Convert to descriptions
       ├─► Batch embed + ingest (500/batch)
       ├─► Load blog JSON (17 articles)
       ├─► Batch embed + ingest (100/batch)
       ├─► Validate with test queries
       └─► Generate summary JSON
       
       ▼
   SUCCESS!
```

### Notebook Approach:
```
Jupyter Notebook
       │
       ▼
Cell 1:  Imports & Setup
Cell 2:  Load Player Data
Cell 3:  Convert to Text
Cell 4:  Initialize Embedding
Cell 5:  Initialize ChromaDB
Cell 6:  Batch Embed Players
Cell 7:  Load Blog Articles
Cell 8:  Batch Embed Blogs
Cell 9:  Validate Search
Cell 10: Generate Summary
       │
       ▼
   SUCCESS!
```

---

##  Module Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TactIQ Project Structure                 │
└─────────────────────────────────────────────────────────────┘

script/
├── day2_embed_and_ingest.py  ◄── Main executable script
│   ├── imports: src.embeddings
│   ├── imports: src.database
│   └── imports: src.text_converter (implicit)

src/
├── embeddings.py              ◄── EmbeddingPipeline class
│   ├── __init__(model_name)
│   ├── embed_text(text) → vector
│   └── embed_batch(texts) → vectors
│
├── database.py                ◄── VectorDatabase class
│   ├── __init__(persist_dir, collection)
│   ├── add_documents(docs, metas, ids)
│   ├── add_documents_batch(...)
│   └── query(query_text, n_results) → results
│
└── text_converter.py          ◄── TableToTextConverter
    ├── create_player_description(row) → text
    ├── create_team_description(row) → text
    └── convert_player_stats(df) → [texts]

notebooks/
└── day2_embedding_vectordb.ipynb  ◄── Interactive version
    (Uses same src/ modules)

data/
├── processed/
│   ├── player_stats_unified_CLEANED.csv  ◄── Input
│   └── day2_summary_*.json               ◄── Output
└── blogs/
    └── tactical_blogs_*.json             ◄── Input

db/
└── chroma/                    ◄── Vector database storage
    ├── chroma.sqlite3
    └── [binary embedding files]
```

---

##  Data Transformation

### Player Stats Example:

```
┌─────────────────────────────────────────────────────────────┐
│  CSV ROW (Structured)                                       │
└─────────────────────────────────────────────────────────────┘
Player: Erling Haaland
Age: 24
Pos: FW
Squad: Manchester City
Comp: Premier League
Gls: 36
Ast: 8
90s: 35.0
market_value_eur: 180000000
Nation: Norway
Season: 2023-2024

              │  create_player_description()
              ▼

┌─────────────────────────────────────────────────────────────┐
│  NATURAL LANGUAGE (Text)                                    │
└─────────────────────────────────────────────────────────────┘
"Erling Haaland is a 24 year old FW playing for Manchester 
City in the Premier League, with 36 goals and 8 assists in 
35.0 matches (90s), Market value: €180.0M, Nationality: 
Norway, Season: 2023-2024."

              │  embed_text()
              ▼

┌─────────────────────────────────────────────────────────────┐
│  EMBEDDING (384-dim vector)                                 │
└─────────────────────────────────────────────────────────────┘
[0.0234, -0.1245, 0.3421, ..., -0.0567, 0.2341]  (384 values)

              │  add_documents()
              ▼

┌─────────────────────────────────────────────────────────────┐
│  CHROMADB DOCUMENT                                          │
└─────────────────────────────────────────────────────────────┘
{
  "id": "player_123_1735567890",
  "document": "Erling Haaland is a 24 year old...",
  "embedding": [0.0234, -0.1245, ...],
  "metadata": {
    "type": "player_stats",
    "player_name": "Erling Haaland",
    "age": "24",
    "position": "FW",
    "squad": "Manchester City",
    "goals": "36",
    "assists": "8",
    ...
  }
}
```

### Blog Article Example:

```
┌─────────────────────────────────────────────────────────────┐
│  JSON ENTRY (Structured)                                    │
└─────────────────────────────────────────────────────────────┘
{
  "title": "The Issue of Passivity – MX",
  "source": "spielverlagerung.com",
  "text": "A solid start was masked by an overly passive 
           5-4-1 and ultimately ended in a 5-1 defeat...",
  "publish_date": "2025-10-25",
  "authors": ["Next Generation"]
}

              │  (No conversion needed)
              ▼

┌─────────────────────────────────────────────────────────────┐
│  TEXT (Already Natural Language)                            │
└─────────────────────────────────────────────────────────────┘
Full article text (~5000 words)

              │  embed_text()
              ▼

┌─────────────────────────────────────────────────────────────┐
│  EMBEDDING (384-dim vector)                                 │
└─────────────────────────────────────────────────────────────┘
[0.1543, 0.2341, -0.0876, ..., 0.1234, -0.3421]  (384 values)

              │  add_documents()
              ▼

┌─────────────────────────────────────────────────────────────┐
│  CHROMADB DOCUMENT                                          │
└─────────────────────────────────────────────────────────────┘
{
  "id": "blog_1_1735567890",
  "document": "A solid start was masked by...",
  "embedding": [0.1543, 0.2341, ...],
  "metadata": {
    "type": "blog_article",
    "title": "The Issue of Passivity – MX",
    "source": "spielverlagerung.com",
    "url": "https://...",
    "publish_date": "2025-10-25",
    ...
  }
}
```

---

##  Semantic Search Flow

```
USER QUERY: "Young striker under 23 with high goals"
     │
     ▼
┌────────────────────────────────┐
│  Embed Query                   │
│  all-MiniLM-L6-v2              │
└────────────────────────────────┘
     │
     ▼ [Query Vector: 384-dim]
┌────────────────────────────────┐
│  ChromaDB Similarity Search    │
│  Cosine similarity vs all docs │
└────────────────────────────────┘
     │
     ▼
┌────────────────────────────────┐
│  Top-K Results (k=5)           │
│  Sorted by similarity          │
└────────────────────────────────┘
     │
     ▼
RESULTS:
1. Erling Haaland (similarity: 0.847)
2. Victor Osimhen (similarity: 0.821)
3. Julian Alvarez (similarity: 0.805)
4. Bukayo Saka (similarity: 0.789)
5. Marcus Rashford (similarity: 0.771)
```

---

##  Performance Metrics

```
┌──────────────────────────────────────────────────────────────┐
│                  EXPECTED PERFORMANCE                        │
└──────────────────────────────────────────────────────────────┘

EMBEDDING SPEED:
├─ CPU:    50-100 docs/s
├─ GPU:    200-500 docs/s
└─ Batch:  Better throughput

TOTAL TIME (19,405 docs):
├─ CPU:    ~5-10 minutes
├─ GPU:    ~2-3 minutes
└─ Breakdown:
    ├─ Load data:      10-20s
    ├─ Convert:        5-10s
    ├─ Embed players:  3-5 min
    ├─ Embed blogs:    10-20s
    └─ Validate:       5-10s

DATABASE SIZE:
├─ ChromaDB:     ~100-150 MB
├─ Embeddings:   ~60-80 MB
└─ Metadata:     ~20-40 MB

QUERY LATENCY:
├─ Single query: 50-200 ms
└─ Batch (5):    100-500 ms
```

---

##  Validation Checklist

```
 BEFORE RUNNING:
   ☐ Python 3.10.x installed
   ☐ In project root directory
   ☐ data/processed/player_stats_unified_CLEANED.csv exists
   ☐ data/blogs/tactical_blogs_*.json exists (optional)
   
 AFTER RUNNING:
   ☐ db/chroma/ directory created
   ☐ db/chroma/ contains ~100-150 MB data
   ☐ data/processed/day2_summary_*.json created
   ☐ logs/day2_ingestion_*.log created
   ☐ Terminal shows " DAY 2 TASKS COMPLETED SUCCESSFULLY!"
   
 VALIDATION:
   ☐ Run: python -c "from src.database import VectorDatabase; db = VectorDatabase(); print(db.collection.count())"
   ☐ Output: ~19,405 documents
   ☐ Test query returns relevant results
   ☐ Both player AND blog results appear
```

---

##  Quick Command Reference

```bash
# 1. STANDARD RUN (Recommended)
python script/day2_embed_and_ingest.py

# 2. RESET DATABASE
python script/day2_embed_and_ingest.py --reset-db

# 3. SMALLER BATCHES (If OOM)
python script/day2_embed_and_ingest.py --batch-size 250

# 4. SKIP BLOGS (Player stats only)
python script/day2_embed_and_ingest.py --skip-blogs

# 5. VERIFY DATABASE
python -c "from src.database import VectorDatabase; db = VectorDatabase(); print(f'Documents: {db.collection.count()}')"

# 6. TEST QUERY
python -c "from src.database import VectorDatabase; db = VectorDatabase(); results = db.query('young striker', n_results=3); print(results['metadatas'])"

# 7. USE NOTEBOOK INSTEAD
jupyter notebook notebooks/day2_embedding_vectordb.ipynb
```

---

##  Technology Stack Visual

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                         │
└─────────────────────────────────────────────────────────────┘

APPLICATION LAYER:
├── script/day2_embed_and_ingest.py  (Python 3.10.x)
└── notebooks/day2_embedding_vectordb.ipynb

                    │
                    ▼

FRAMEWORK LAYER:
├── pandas (Data processing)
├── loguru (Logging)
└── json, time, datetime (Utilities)

                    │
                    ▼

CORE LIBRARIES:
├── src/embeddings.py
│   └── sentence-transformers (HuggingFace)
│       └── all-MiniLM-L6-v2 (384-dim)
│
├── src/database.py
│   └── chromadb (Vector database)
│       ├── Persistent client
│       └── Local storage
│
└── src/text_converter.py
    └── pandas (DataFrame processing)

                    │
                    ▼

DATA LAYER:
├── CSV files (Structured data)
├── JSON files (Unstructured text)
└── ChromaDB storage (Binary vectors)
```

---

##  Learning Path

```
BEGINNER:
   Start here ──► Notebook (step-by-step)
                  │
                  ▼
                  Understand each cell
                  │
                  ▼
                  Run validation queries
                  
INTERMEDIATE:
   Script execution ──► Understand parameters
                        │
                        ▼
                        Modify batch sizes
                        │
                        ▼
                        Test custom data

ADVANCED:
   Modify src/ modules ──► Add new converters
                           │
                           ▼
                           Custom embeddings
                           │
                           ▼
                           Production deployment
```

---

##  Summary

```
┌─────────────────────────────────────────────────────────────┐
│  DAY 2 COMPLETE: EMBEDDING & VECTOR DATABASE SETUP          │
└─────────────────────────────────────────────────────────────┘

INPUT:       19,388 player records + 17 blog articles
PROCESS:     Convert → Embed → Store
OUTPUT:      19,405 semantic vectors in ChromaDB
TIME:        5-10 minutes
STORAGE:     ~100-150 MB

 Reproducible:  Single command execution
 Flexible:      CLI arguments + notebook
 Modular:       Reusable src/ components
 Validated:     Test queries working
 Documented:    Comprehensive guides

READY FOR DAY 3: Agent Framework & CRAG Implementation
```

---

**Execute Now:**
```bash
python script/day2_embed_and_ingest.py
```
