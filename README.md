# Indian Commercial Court RAG - Corpus Foundation

This repository contains the corpus and vector-search foundation for an Indian
Commercial Court legal RAG system. It normalizes heterogeneous judgment records,
deduplicates and stores cases and paragraphs in PostgreSQL, and indexes paragraph
embeddings in Qdrant.

The current scope deliberately excludes a frontend, LLM answer generation,
LangChain, OpenSearch, metadata filtering, reranking, OCR, and authentication.

## Repository layout

```text
backend/legal_rag/
  config.py                 Environment-backed runtime settings
  database.py               SQLAlchemy engine, sessions, and test schema helper
  models.py                 Case, Paragraph, and Statute models
  schema_migrations.py      Safe empty/legacy/versioned database upgrades
  acquisition.py            Resumable official-PDF download and validation
  extraction.py             Resumable PDF-to-canonical-JSONL conversion
  corpus_audit.py           Relational quality and vector-coverage metrics
  corpus/                   Canonical schema, normalization, hashing, extraction
  services/ingestion.py     Transactional case/paragraph insertion
  embeddings/               Embedding interface and Sentence Transformers adapter
  retrieval/                Dense, BM25, and rank-fused paragraph retrieval
  vector/                   Qdrant paragraph index adapter
scripts/
  download_judgments.py     Pilot judgment PDF acquisition and audit
  extract_judgments.py      Text-native PDF extraction with page boundaries
  audit_corpus.py           PostgreSQL quality and Qdrant identity audit
  migrate_database.py       Validated legacy-to-Alembic upgrade command
  ingest_corpus.py          Resumable JSONL ingestion
  index_vectors.py          PostgreSQL-to-Qdrant paragraph indexing
  test_search.py            Semantic-search command line tool
  test_bm25.py              BM25 lexical-search command line tool
tests/                       Corpus and database unit tests
docker-compose.yml          PostgreSQL and Qdrant services
migrations/                 Alembic environment and ordered schema revisions
data/manifests/             Tracked pilot provenance manifest and audit
```

## Prerequisites

- Python 3.10 or newer
- Docker with Docker Compose

## Setup

Create the local environment file. Docker Compose reads `.env` automatically;
the checked-in defaults also work without it.

```powershell
Copy-Item .env.example .env
```

Start PostgreSQL and Qdrant:

```powershell
docker compose up -d
docker compose ps
```

Both rows should report `healthy`. Direct checks are also available:

```powershell
docker compose exec -T postgres pg_isready -U legal_rag -d legal_rag
Invoke-WebRequest -UseBasicParsing http://localhost:6333/healthz
```

The Compose ports bind to `127.0.0.1` only. The example credentials and
unauthenticated local Qdrant service are development defaults and must not be
exposed or reused in a production deployment.

Create a virtual environment and install the package with test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The Python scripts read configuration from process environment variables and
otherwise use the values shown in `.env.example`. If you change Compose database
credentials, export the matching `DATABASE_URL` before running a script. For
example:

```powershell
$env:DATABASE_URL = "postgresql+psycopg://legal_rag:legal_rag_dev_password@127.0.0.1:5432/legal_rag"
```

## Database migrations

For a new empty database, or a database already managed by Alembic, upgrade to
the latest schema with:

```powershell
alembic upgrade head
```

Databases created by the earlier `Base.metadata.create_all()` implementation have
the three corpus tables but no `alembic_version` table. Back up that database,
then use the guarded upgrade command instead of manually stamping it:

```powershell
python scripts/migrate_database.py
```

The command first requires the exact legacy `cases`, `paragraphs`, and `statutes`
tables, columns, types, nullability, primary keys, indexes,
document/paragraph uniqueness, and cascading case foreign key. It then runs the
baseline revision against that validated schema, adds and deterministically
backfills `paragraph_uid`, enforces non-null/global uniqueness, and advances to
head. Unknown or partially matching schemas are refused; never use
`alembic stamp` to bypass that validation.

Ingestion runs this same safe upgrade automatically before processing JSONL. If
the old database already had a Qdrant index containing numeric point IDs, rebuild
it once after migration so those stale points cannot coexist with UUID points:

```powershell
python scripts/index_vectors.py --recreate
```

## Canonical case format

The ingestion boundary converts every source record to this representation:

| Field | Required | Notes |
| --- | --- | --- |
| `title` | No | Falls back to `case_number`, then `Untitled case` |
| `case_number` | No | Source court/docket identifier |
| `court` | No | Court or tribunal name |
| `judgment_date` | No | ISO dates and common Indian date formats are accepted |
| `source` | No | Can also be supplied with `--source` |
| `source_url` | No | Original document URL |
| `raw_text` | Yes | Normalized before hashing and storage |
| `document_hash` | Computed | SHA256 of normalized `raw_text` |

The normalizer also recognizes common aliases such as `case_title`, `case_name`,
`case_no`, `court_name`, `decision_date`, `url`, `judgment_text`, `full_text`, and
`content`. It searches common nested containers including `metadata`, `details`,
`document`, `case`, and `data`.

## Pilot judgment acquisition

The first pilot corpus is defined by the tracked manifest at
`data/manifests/judgments_pilot.jsonl`. Its records identify 20 commercial-law
judgments hosted directly by the Supreme Court of India and 80 additional
judgments from the public Supreme Court judgment archive generated from the
eCourts/Supreme Court Reports corpus. Every record includes the official SCR or
judgment-search provenance page, exact PDF URL, retrieval time, local filename,
SHA-256, byte size, and detected MIME type. `document_type` is required to be
`judgment`, and the PDF text must contain an explicit judgment heading. This
100-case, Supreme-Court-only set is deliberately a pilot; later corpus rounds
must add representative High Court commercial-division and Commercial Court
judgments.

The archive is the CC-BY-4.0
[Indian Supreme Court Judgments public dataset](https://github.com/vanga/indian-supreme-court-judgments),
which documents its eCourts acquisition and publishes matching metadata. It is
a mirror, not an official court host; the manifest therefore retains the
official SCR search page separately and the downloader trusts only this exact
archive hostname.

Download or validate the pilot corpus with:

```powershell
python scripts/download_judgments.py
```

PDFs are stored under `data/raw/judgments/`, which is Git-ignored. The manifest
is the resume checkpoint: an existing file is parsed, re-hashed, and reused
without another HTTP request. New responses are accepted only when they come
from an HTTPS `.gov.in` or `.nic.in` host or the single allow-listed public
Supreme Court judgment archive host. They must have PDF magic, open with a PDF
parser, contain pages and at least 200 extractable non-whitespace characters,
and have a corpus-unique hash. Downloads use explicit timeouts, bounded retries,
and exponential backoff. Files are placed atomically only after validation.

The latest categorized audit is written to
`data/manifests/judgments_pilot_audit.json`. Individual errors are appended to
the Git-ignored `data/failed/judgment_downloads.jsonl`. CAPTCHA-protected search
interfaces are not automated; unavailable sources remain failures for manual
review.

Convert the locally validated PDFs to canonical ingestion JSONL with:

```powershell
python scripts/extract_judgments.py --restart
python scripts/ingest_corpus.py data/processed/judgments_pilot.jsonl --restart
```

Extraction uses `pypdf` only and does not perform OCR. It rejects encrypted,
corrupt, empty, and non-text PDFs, verifies each file against the acquisition
manifest, and joins page text with a form-feed (`\f`) boundary. The canonical
records contain only `title`, `case_number`, `court`, `judgment_date`, `source`,
`source_url`, and `raw_text`; `source_url` retains the exact direct PDF URL from
the acquisition manifest, while the acquisition manifest's `source_page_url`
preserves official provenance.
Processed JSONL, extraction checkpoints, and failure logs are Git-ignored. The
tracked aggregate audit is
`data/manifests/judgments_pilot_extraction_audit.json`. Without `--restart`,
completed PDFs whose manifest hashes match the checkpoint are safely reused.

## Pilot corpus audit and vector index

Generate relational quality metrics before indexing, rebuild Qdrant from the
authoritative PostgreSQL paragraphs, and then verify exact identity coverage:

```powershell
python scripts/audit_corpus.py --skip-qdrant
python scripts/index_vectors.py --recreate
python scripts/audit_corpus.py
```

The tracked report is
`data/manifests/judgments_pilot_corpus_audit.json`. It includes case/paragraph
counts, nearest-rank p95 and other per-case paragraph statistics, missing
metadata, empty and sub-20-character paragraph counts, duplicate hashes/UUIDs,
page-number coverage, acquisition/extraction outcomes, and a full PostgreSQL
`paragraph_uid` versus Qdrant point-ID comparison including indexing coverage.
`--recreate` is destructive only to the configured Qdrant collection and is the
supported way to remove stale points before a rebuild.

Run the pilot semantic sanity queries with:

```powershell
python scripts/test_search.py "unilateral appointment of arbitrator" --top-k 10
python scripts/test_search.py "commercial wisdom of committee of creditors" --top-k 10
python scripts/test_search.py "Section 7 IBC admission discretion" --top-k 10
```

These three searches are smoke checks, not a formal retrieval evaluation. This
100-case, Supreme-Court-only pilot cannot measure production recall, ranking
quality, court coverage, or temporal coverage. Representative High Court
commercial-division and Commercial Court judgments, followed by labeled query
relevance judgments, are required before retrieval quality can be evaluated.

## Ingest a JSONL corpus

Input must contain one JSON object per line. A minimal record looks like:

```json
{"title":"Acme Pvt. Ltd. v. Zenith Ltd.","case_number":"CS(COMM) 42/2024","court":"Delhi High Court","judgment_date":"2024-05-16","source":"court-portal","source_url":"https://example.test/judgments/42","raw_text":"1. The suit concerns a commercial contract.\n\n2. The claim is decreed."}
```

Run ingestion with:

```powershell
python scripts/ingest_corpus.py .\data\cases.jsonl --source court-portal
```

Each line runs in its own database transaction. Duplicate normalized document
hashes are skipped. A bad line does not stop later lines: its line number, error,
and source record are appended to `data/failed/*.jsonl`.

Progress is atomically saved after every processed line in
`data/checkpoints/*.json`. Running the same command again resumes after the last
processed line. The checkpoint includes the input file's SHA256 identity, so a
replaced or edited file cannot silently reuse stale line progress. Use `--restart`
to scan a changed file from line one again; database hash constraints still
prevent duplicate cases. Checkpoints are also scoped to the database URL and
verify that their most recently persisted document still exists, preventing a
cleared or different database from being mistaken for a completed ingestion.
Failure logs remain append-only when `--restart` is used, preserving the earlier
error history.

Additional options:

```powershell
python scripts/ingest_corpus.py --help
```

## Index paragraphs in Qdrant

After ingestion, embed all stored paragraphs and upsert them by stable paragraph
ID:

```powershell
python scripts/index_vectors.py --batch-size 64
```

The default embedding model is downloaded by Sentence Transformers on first use.
Every Qdrant point stores exactly this payload:

```text
case_id, paragraph_uid, title, case_number, court, judgment_date, source_url,
year, paragraph_number, page_number, text
```

Rerunning the indexer is idempotent because the deterministic UUIDv5
`paragraph_uid` is the Qdrant point ID. The numeric PostgreSQL `Paragraph.id` is
used only for efficient keyset pagination while indexing; it is not a durable
citation or vector identity. Normal runs do not delete points, so after deleting
database paragraphs or resetting PostgreSQL, rebuild an exact collection with
`python scripts/index_vectors.py --recreate`. The destructive reset happens only
when this flag is supplied. If `EMBEDDING_MODEL` is changed, set its matching
`EMBEDDING_DIMENSION` and use a new `QDRANT_COLLECTION`; an incompatible existing
collection is rejected instead of overwritten.

## Dense, BM25, and hybrid search

Print the top five semantically similar legal paragraphs:

```powershell
python scripts/test_search.py "breach of an arbitration agreement" --top-k 5
```

Use `--model` or `--collection` to override those settings for one run.
Search requires an existing compatible collection and will direct you to run the
indexer instead of silently creating an empty one.

Build an in-memory BM25 index from the authoritative PostgreSQL paragraphs and
run a lexical query with:

```powershell
python scripts/test_bm25.py "Section 7 IBC admission discretion" --top-k 5
```

BM25 uses deterministic NFKC normalization, Unicode-aware case-folded word and
number tokens, and standard Okapi scoring with term-frequency saturation,
document-frequency IDF, and document-length normalization. It does not mutate
stored paragraph text. The CLI reports index-build and query timing on each run.

Dense and BM25 retrieval expose the same typed paragraph result fields, including
durable `paragraph_uid`, rank, native retrieval score, case metadata, paragraph
and page numbers, and source URL. Both remain independently usable.

Hybrid search retrieves up to 50 candidates independently from each retriever
and combines their rank positions with Reciprocal Rank Fusion (RRF), using the
conventional default `k=60`:

```powershell
python scripts/test_hybrid.py "commercial wisdom of committee of creditors" --top-k 10
```

Use `--bm25-depth`, `--dense-depth`, or `--rrf-k` to override those values for a
run. RRF never mixes or normalizes raw BM25 and cosine scores; the hybrid output
retains each native rank and score only as provenance alongside the final RRF
score. Results are deduplicated by durable `paragraph_uid`. Metadata filtering
and cross-encoder reranking are not implemented yet.

## Tests

Run the complete suite:

```powershell
python -m pytest -q
```

The tests cover normalization, document and paragraph deduplication, paragraph
number/page extraction, relational insertion, duplicate skipping, statute
storage, resumable failure-tolerant ingestion, checkpoint safeguards, and Qdrant
collection/payload behavior. Migration tests upgrade both empty and legacy
temporary databases and verify that migration/runtime UUID generation agrees.
Mocked acquisition tests cover retries, resume behavior, PDF validation,
deduplication, failure categories, allow-listed archive provenance, and the
tracked 100-record manifest contract.
Extraction tests use generated in-memory PDFs and cover form-feed page
boundaries, atomic resume, metadata failures, and rejection of corrupt,
encrypted, and non-text PDFs; tests never download the real pilot files.
Corpus-audit tests use a temporary SQLite database and fake Qdrant pagination;
they do not require Docker or download an embedding model.
BM25 tests use only synthetic in-memory paragraph fixtures and cover lexical
relevance, document frequency, term-frequency saturation, shared dense/lexical
result fields, validation, deduplication, and deterministic tie ordering.
Hybrid tests use synthetic retriever outputs and cover exact RRF mathematics,
cross-list boosting, one-list candidates, UID deduplication, deterministic ties,
validation, candidate depths, and rank/native-score provenance without Docker or
an embedding download.
SQLite is used for isolated database tests; the production connection remains
PostgreSQL.

## Data model

- `cases`: canonical judgment metadata, normalized raw text, unique document hash,
  and creation timestamp.
- `paragraphs`: case foreign key, paragraph/page numbers, normalized text, and a
  per-case unique text hash. Its numeric `id` is a local database surrogate;
  `paragraph_uid` is the globally unique, insertion-order-independent UUID used
  for durable citations and Qdrant. Deleting a case cascades to its paragraphs.
- `statutes`: act name, section, title, and statutory text.

Stop the local services without deleting their named volumes:

```powershell
docker compose down
```
