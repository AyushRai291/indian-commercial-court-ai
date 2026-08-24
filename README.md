# Indian Commercial Court RAG - Corpus Foundation

This repository contains the corpus, retrieval, and grounded-generation foundation
for an Indian Commercial Court legal RAG system. It normalizes heterogeneous
judgment records, stores cases and paragraphs in PostgreSQL, indexes paragraph
embeddings in Qdrant, and can generate answers over request-local reranked evidence.

The Day 14 frontend is a presentation shell driven by clearly labelled static mock
responses. Live frontend/API integration, claim-level citation verification,
LangChain, OpenSearch, OCR, and authentication remain out of scope.

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
  retrieval/                Dense, BM25, rank-fused, and reranked retrieval
  evaluation/               Gold validation and paragraph-level retrieval metrics
  vector/                   Qdrant paragraph index adapter
  api/                      FastAPI schemas, shared retrieval service, and app
  generation/               Evidence IDs, grounded prompt, provider, and answer service
frontend/                   React/Vite legal-research presentation shell
scripts/
  download_judgments.py     Pilot judgment PDF acquisition and audit
  extract_judgments.py      Text-native PDF extraction with page boundaries
  audit_corpus.py           PostgreSQL quality and Qdrant identity audit
  migrate_database.py       Validated legacy-to-Alembic upgrade command
  ingest_corpus.py          Resumable JSONL ingestion
  index_vectors.py          PostgreSQL-to-Qdrant paragraph indexing
  evaluate_retrieval.py     Frozen four-system retrieval benchmark
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

## Search and grounded-answer API

Start the API after PostgreSQL and Qdrant are healthy and the package is installed:

```powershell
uvicorn legal_rag.api.app:app --reload
```

The generated OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.
`GET /health` is a lightweight process-liveness check. `POST /search` accepts
`bm25`, `dense`, `hybrid`, and `reranked` modes; `reranked` and `top_k=10` are
the defaults, and `top_k` must be between 1 and 50. The hybrid stages use the
measured 50 BM25 candidates, 50 dense candidates, and RRF `k=10`; reranking
scores the first 30 hybrid candidates. A reranked request above 30 results can
therefore return at most 30 without changing that frozen candidate depth.

Default reranked search:

```powershell
$body = @{
  query = "commercial wisdom of committee of creditors"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" -Body $body
```

BM25 search:

```powershell
$body = @{
  query = "unilateral appointment of arbitrator"
  retrieval_mode = "bm25"
  top_k = 10
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" -Body $body
```

Filtered search (filters use exact normalized AND matching):

```powershell
$body = @{
  query = "commercial wisdom of committee of creditors"
  retrieval_mode = "hybrid"
  filters = @{
    court = "Supreme Court of India"
    year = 2019
  }
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType "application/json" -Body $body
```

Every hit returns the durable paragraph UID, unchanged paragraph text, canonical
case metadata, source URL, paragraph and page numbers, available native
BM25/dense ranks and scores, RRF score and hybrid rank, cross-encoder score, and
final rank. Fields unavailable for the selected retrieval mode are consistently
`null`; no answer or citation text is generated.

`POST /answer` always invokes the existing reranked search path internally. It
assigns request-local IDs (`E1`, `E2`, ...) in reranked order, passes only that
evidence to the configured model, and returns the answer plus the complete
paragraph provenance needed by the UI. Callers cannot provide evidence or select
a weaker retrieval mode.

Configure the single OpenAI provider through environment variables; never commit a
real key:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "gpt-5-mini"       # optional default
$env:OPENAI_TIMEOUT_SECONDS = "60"     # optional default
```

Example request:

```powershell
$body = @{
  query = "Can an ineligible arbitrator nominate another person as arbitrator?"
  top_k = 10
  filters = @{
    court = "Supreme Court of India"
    year = $null
    case_number = $null
  }
} | ConvertTo-Json -Depth 3
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/answer `
  -ContentType "application/json" -Body $body
```

The response shape is:

```json
{
  "query": "...",
  "answer": "... [E1] ...",
  "used_evidence_ids": ["E1"],
  "evidence": [
    {
      "evidence_id": "E1",
      "paragraph_uid": "...",
      "case_name": "...",
      "reranked_rank": 1,
      "page_number": 42,
      "paragraph_number": 118,
      "source_url": "...",
      "text": "..."
    }
  ],
  "retrieval_latency_ms": 0,
  "generation_latency_ms": 0,
  "total_latency_ms": 0
}
```

The provider uses a structured output contract and the API rejects unknown or
malformed evidence IDs and mismatches between inline citations and
`used_evidence_ids`. These are structural integrity checks only: Day 14 does not
classify claims as supported, partial, or unsupported. With zero retrieved
evidence, the provider is skipped and a deterministic insufficiency response is
returned.

## Frontend presentation shell

The desktop-first legal research workspace currently uses API-shaped fixtures in
`frontend/src/mocks/`; it does not call `/answer` or write demo records to the
corpus. Run it with:

```powershell
Set-Location frontend
npm install
npm run dev
```

The shell includes search and filter controls, a grounded-answer view, clickable
`[E1]` citations, ranked evidence cards, a full-provenance evidence panel, and
loading, empty, no-result, backend-error, and generation-error states.

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
year, court_filter, case_number_filter, paragraph_number, page_number, text
```

Collections created before metadata-filter support lack the two normalized
helper fields. Rebuild such a collection once with
`python scripts/index_vectors.py --recreate` before running filtered searches.

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
and combines their rank positions with Reciprocal Rank Fusion (RRF). The measured
pilot default is `k=10`:

```powershell
python scripts/test_hybrid.py "commercial wisdom of committee of creditors" --top-k 10
```

Use `--bm25-depth`, `--dense-depth`, or `--rrf-k` to override those values for a
run. RRF never mixes or normalizes raw BM25 and cosine scores; the hybrid output
retains each native rank and score only as provenance alongside the final RRF
score. Results are deduplicated by durable `paragraph_uid`.

BM25, dense, and hybrid search accept the same optional exact metadata filters:

```powershell
python scripts/test_hybrid.py "commercial wisdom of committee of creditors" `
  --court "Supreme Court of India" --year 2019 --top-k 10
python scripts/test_bm25.py "arbitration" `
  --case-number "Arbitration Application No. 32 of 2019" --top-k 5
python scripts/test_search.py "insolvency admission" --year 2022 --top-k 5
```

`--court`, `--year`, and `--case-number` combine with AND semantics. Court and
case-number matching uses Unicode NFKC normalization, case-folding, and collapsed
whitespace while preserving canonical metadata for display. Filters constrain
the eligible BM25 and native Qdrant candidate sets before RRF; they do not boost,
normalize, or otherwise modify relevance scores.

Cross-encoder reranking takes the first 30 hybrid RRF candidates, scores each
unchanged `(query, paragraph text)` pair in batches with
`cross-encoder/ms-marco-MiniLM-L6-v2`, and returns the best 10 by native model
score:

```powershell
python scripts/test_rerank.py "commercial wisdom of committee of creditors" `
  --candidate-k 30 --top-k 10
```

The model loads lazily on the first reranked search and is reused by subsequent
searches on the same reranker. Configure the candidate count, final count, model,
and CPU-friendly batch size with `RERANKER_CANDIDATE_K`, `RERANKER_TOP_K`,
`RERANKER_MODEL`, and `RERANKER_BATCH_SIZE`, or the corresponding CLI flags.
`--court`, `--year`, and `--case-number` reuse the existing pre-retrieval filters;
only eligible hybrid candidates reach the cross-encoder. Output preserves native
BM25/dense provenance, RRF score and hybrid rank alongside the cross-encoder
score and final reranked rank. Formal pilot metrics and tuning evidence are
documented below.

## Gold retrieval queries

The tracked gold set at `data/evaluation/gold_queries.jsonl` contains 40
realistic legal-research queries grounded in verified paragraphs from the
100-judgment corpus. Each paragraph is identified by durable `paragraph_uid`
and graded `3` for a direct answer or central holding, `2` for strongly relevant
support, or `1` for useful context. A review-friendly rendering with short,
database-sourced evidence snippets is tracked at
`data/evaluation/gold_queries_review.md`.

Validate structure, uniqueness, relevance requirements, and exact PostgreSQL
metadata/provenance with:

```powershell
python scripts/validate_gold_queries.py data/evaluation/gold_queries.jsonl
```

This command prints dataset-characterization statistics only and does not run
retrieval.

## Retrieval evaluation

Run the frozen 40-query pilot benchmark over the complete corpus with no metadata
filters:

```powershell
python scripts/evaluate_retrieval.py
python scripts/evaluate_retrieval.py --validate-only
```

The Day 11 runner builds one BM25 index and reuses one embedding model, Qdrant
client, and cross-encoder across all queries. It preserves the historical BM25
top 10; dense top 10; RRF hybrid with 50 candidates per branch, `k=60`, and final
top 10; and cross-encoder reranking of 50 hybrid candidates to a final top 10.

Recall@5 and Recall@10 use every positive gold grade as binary relevance and are
macro-averaged. MRR is the macro reciprocal rank of the first exact gold
`paragraph_uid` in the returned top 10. nDCG@10 uses the gold grades with gain
`2^relevance - 1`, `log2(rank + 1)` discount, and macro averaging. Same-case or
similar-text results receive no credit without an exact paragraph UID match.

Tracked outputs are written to:

- `data/evaluation/results/retrieval_metrics.json`
- `data/evaluation/results/retrieval_per_query.jsonl`
- `data/evaluation/results/retrieval_evaluation_report.md`

These are the immutable pre-tuning measurements on the frozen pilot gold set.

## Retrieval tuning and error analysis

Run or validate the fixed Day 12 ablation grid with:

```powershell
python scripts/tune_retrieval.py
python scripts/tune_retrieval.py --validate-only
```

The runner caches native BM25/dense top-50 rankings once, tests RRF
`k={10,20,40,60,80,100}`, controlled 30/40/50 symmetric and asymmetric native
depths, and cross-encoder candidate depths 30/40/50. It records every tested
configuration under `data/evaluation/tuning/`:

- `retrieval_ablation_results.json`
- `retrieval_error_analysis.json`
- `retrieval_tuning_report.md`

The measured defaults are 50 BM25 candidates, 50 dense candidates, RRF `k=10`,
and 30 reranker candidates, with final top 10 unchanged. The tuned reranker
improved the original reranker from nDCG@10 `0.292882` to `0.297160`, but remained
below the BM25 baseline `0.303318`; BM25 therefore remains the best pilot system
on Recall@5, MRR, and nDCG@10. Exact-UID candidate generation and cross-encoder
ordering both remain measured failure sources. These results cover only 40
queries and a 100-judgment Supreme-Court-only pilot; no significance claim is
made, and paragraph splitting was not changed because it would invalidate the
frozen paragraph-UID labels.

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
Metadata-filter tests cover deterministic normalization, AND semantics, full-set
BM25 eligibility, native Qdrant filter forwarding, hybrid propagation, empty
matches, and unfiltered ranking regressions.
Cross-encoder tests use fakes only and cover candidate batching, native-score
ordering, provenance, deterministic ties, limits, filters, duplicate safety,
empty retrieval, lazy loading, and model reuse without downloading model files.
Evaluation tests use synthetic rankings only and verify exact-UID Recall@5/10,
top-10 MRR, graded nDCG@10, duplicate safety, macro averaging, and artifact
completeness/recomputation without calling PostgreSQL, Qdrant, or either model.
Tuning tests cover the fixed experiment grid, candidate recall, category
aggregation, reranker failure classification, deterministic selection, and
artifact validation without running real models.
Gold-query validator tests use synthetic SQLite fixtures and cover duplicate
identities, dangling UIDs, metadata mismatch, grading rules, duplicate labels,
empty queries, statistics, and review rendering without running retrieval.
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
