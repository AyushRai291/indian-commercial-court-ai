# Retrieval Evaluation Report

Generated: `2026-08-22T07:51:48.115041+00:00`

This benchmark evaluates all 40 frozen pilot queries over the complete 100-judgment corpus without metadata filters. No retrieval parameter or gold label was tuned during evaluation.

## Metric definitions

- Recall@K is macro-averaged binary paragraph recall over all positive gold grades, with duplicate UIDs counted once.
- MRR is the macro mean reciprocal rank of the first exact gold UID in the returned top-10 ranking; misses score zero.
- nDCG@10 is macro-averaged graded nDCG with gain `2^relevance - 1` and discount `log2(rank + 1)`.
- Relevance requires an exact `paragraph_uid` match.

## Configuration

- Corpus paragraphs: 18822
- Gold queries: 40
- BM25: k1=1.5, b=0.75
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`
- Hybrid: BM25 depth=50, dense depth=50, RRF k=60
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`, candidate depth=50
- Final depth: 10
- Metadata filters: none

## Aggregate comparison

| Retrieval System | Recall@5 | Recall@10 | MRR | nDCG@10 |
| ---------------- | -------: | --------: | --: | ------: |
| BM25 | 0.277083 | 0.343750 | 0.377738 | 0.303318 |
| Dense | 0.154167 | 0.204167 | 0.212708 | 0.175465 |
| BM25 + Dense + RRF | 0.214583 | 0.293750 | 0.316905 | 0.250746 |
| Hybrid + Reranker | 0.241667 | 0.352083 | 0.361319 | 0.292882 |

## Factual diagnostics

- BM25: relevant hit in top 5 for 25 queries; top 10 for 29; zero hits in top 10 for 11; rank-1 hit for 9. Best nDCG query IDs: Q032. Worst: Q003, Q004, Q010, Q012, Q015, Q018, Q021, Q022, Q023, Q025, Q030.
- Dense: relevant hit in top 5 for 11 queries; top 10 for 15; zero hits in top 10 for 25; rank-1 hit for 6. Best nDCG query IDs: Q002. Worst: Q003, Q004, Q005, Q006, Q007, Q008, Q010, Q011, Q012, Q013, Q014, Q015, Q017, Q019, Q022, Q023, Q024, Q025, Q029, Q030, Q031, Q033, Q036, Q038, Q040.
- BM25 + Dense + RRF: relevant hit in top 5 for 19 queries; top 10 for 24; zero hits in top 10 for 16; rank-1 hit for 9. Best nDCG query IDs: Q016. Worst: Q003, Q004, Q006, Q007, Q010, Q011, Q012, Q015, Q017, Q018, Q019, Q021, Q022, Q023, Q025, Q040.
- Hybrid + Reranker: relevant hit in top 5 for 21 queries; top 10 for 26; zero hits in top 10 for 14; rank-1 hit for 9. Best nDCG query IDs: Q015. Worst: Q003, Q004, Q005, Q006, Q007, Q010, Q017, Q018, Q021, Q022, Q023, Q030, Q036, Q040.

Strict nDCG@10 comparisons:

- Hybrid beats both individual retrievers: 6 queries (Q001, Q016, Q028, Q029, Q030, Q034).
- Reranker improves hybrid: 13 queries (Q002, Q008, Q011, Q012, Q014, Q015, Q019, Q024, Q025, Q029, Q033, Q038, Q039).
- Reranker worsens hybrid: 12 queries (Q001, Q005, Q016, Q020, Q026, Q027, Q030, Q031, Q032, Q034, Q036, Q037).

The reranker can reorder only the 50 hybrid candidates it receives; it cannot recover a relevant paragraph absent from that pool.

## Approximate runtimes

- BM25 evaluation: 3.301 s
- Dense evaluation: 85.230 s
- Hybrid evaluation: 87.620 s
- Reranker evaluation: 267.003 s
- Cross-encoder cold load: 4.490 s
- Query execution total: 443.156 s
- Setup plus execution: 550.173 s

These are pilot measurements, not tuning conclusions. Per-query rankings are retained for later error analysis.
