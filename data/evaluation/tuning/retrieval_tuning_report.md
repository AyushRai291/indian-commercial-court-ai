# Retrieval Tuning and Error Analysis

Generated: `2026-08-22T08:33:33.085445+00:00`

The frozen 40-query gold set and Day 11 metrics were validated before all experiments. BM25 parameters, embedding model, cross-encoder, corpus, paragraph UIDs, and gold labels were unchanged.

## Final comparison

| System | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 baseline | 0.277083 | 0.343750 | 0.377738 | 0.303318 |
| Dense baseline | 0.154167 | 0.204167 | 0.212708 | 0.175465 |
| Original RRF | 0.214583 | 0.293750 | 0.316905 | 0.250746 |
| Original Reranker | 0.241667 | 0.352083 | 0.361319 | 0.292882 |
| Tuned Hybrid | 0.250000 | 0.339583 | 0.341171 | 0.279577 |
| Tuned Hybrid + Reranker | 0.250000 | 0.347917 | 0.366736 | 0.297160 |

## Selection

- Best hybrid: `rrf_b50_d50_k10` — BM25 depth 50, dense depth 50, RRF k=10.
- Best reranker: `rerank_rrf_b50_d50_k10_c30` — candidate depth 30.
- Default decision: defaults changed to the measured selected hybrid/reranker values.
- Selection order: nDCG@10, Recall@10, MRR, runtime, then simpler configuration.
- RRF k=10 improved every aggregate metric over k=60 by concentrating weight on the strongest native ranks, but tuned hybrid nDCG@10 0.279577 still trails BM25 0.303318.
- Dense contributes unique candidate recall (the BM25+dense union reaches 0.718750 at depth 50), but its much weaker ordering receives equal rank-based influence in symmetric RRF and can displace exact BM25 gold hits.
- Reranker depth 30 is the nDCG-selected guardrail. Depths 40/50 slightly raise Recall@10 but lower MRR and nDCG as additional lower-hybrid candidates become eligible for promotion.

## Candidate recall

| Candidate source | Recall@30 | Recall@50 |
|---|---:|---:|
| BM25 | 0.510417 | 0.664583 |
| Dense | 0.347917 | 0.397917 |
| BM25 ∪ Dense | 0.589583 | 0.718750 |
| Baseline RRF | 0.468750 | 0.602083 |

Selected-reranker zero-hit failures: 6 had no gold UID in its hybrid candidate pool; 8 had a candidate available but outside the reranked top 10.

## RRF sweep

| k | BM25 depth | Dense depth | R@5 | R@10 | MRR | nDCG@10 | Cached runtime (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 50 | 50 | 0.2500 | 0.3396 | 0.3412 | 0.2796 | 0.0238 |
| 20 | 50 | 50 | 0.2438 | 0.3063 | 0.3307 | 0.2624 | 0.0423 |
| 40 | 50 | 50 | 0.2313 | 0.2938 | 0.3212 | 0.2528 | 0.0314 |
| 60 | 50 | 50 | 0.2146 | 0.2938 | 0.3169 | 0.2507 | 0.0206 |
| 80 | 50 | 50 | 0.2146 | 0.2938 | 0.3166 | 0.2506 | 0.0268 |
| 100 | 50 | 50 | 0.2146 | 0.3021 | 0.3166 | 0.2522 | 0.0211 |
| 60 | 30 | 30 | 0.2271 | 0.3271 | 0.3261 | 0.2661 | 0.0170 |
| 60 | 40 | 40 | 0.2146 | 0.3063 | 0.3166 | 0.2541 | 0.0185 |
| 60 | 50 | 30 | 0.2146 | 0.3063 | 0.3186 | 0.2560 | 0.2246 |
| 60 | 30 | 50 | 0.2146 | 0.3063 | 0.3197 | 0.2567 | 0.0135 |

## Reranker-depth sweep

| Candidates | R@5 | R@10 | MRR | nDCG@10 | Shared inference (s) |
|---:|---:|---:|---:|---:|---:|
| 30 | 0.2500 | 0.3479 | 0.3667 | 0.2972 | 41.832 |
| 40 | 0.2417 | 0.3521 | 0.3616 | 0.2938 | 60.365 |
| 50 | 0.2417 | 0.3521 | 0.3613 | 0.2929 | 78.460 |

## Day 11 category breakdown

### Query type

| Category | System | N | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| doctrine_or_test | BM25 | 5 | 0.1333 | 0.2000 | 0.1600 | 0.1534 |
| doctrine_or_test | Dense | 5 | 0.1333 | 0.1333 | 0.2000 | 0.1626 |
| doctrine_or_test | RRF | 5 | 0.0667 | 0.1333 | 0.2200 | 0.1397 |
| doctrine_or_test | Reranker | 5 | 0.0667 | 0.1333 | 0.2000 | 0.1426 |
| exact_terminology | BM25 | 6 | 0.3472 | 0.3472 | 0.5972 | 0.4378 |
| exact_terminology | Dense | 6 | 0.1667 | 0.2222 | 0.3542 | 0.2619 |
| exact_terminology | RRF | 6 | 0.2639 | 0.2639 | 0.4083 | 0.2975 |
| exact_terminology | Reranker | 6 | 0.3889 | 0.4306 | 0.4889 | 0.4156 |
| fact_pattern | BM25 | 6 | 0.4444 | 0.4444 | 0.4167 | 0.3826 |
| fact_pattern | Dense | 6 | 0.1944 | 0.2500 | 0.1333 | 0.1790 |
| fact_pattern | RRF | 6 | 0.3611 | 0.3611 | 0.3472 | 0.3252 |
| fact_pattern | Reranker | 6 | 0.3056 | 0.4444 | 0.5185 | 0.3977 |
| legal_principle | BM25 | 4 | 0.2500 | 0.3750 | 0.1750 | 0.1568 |
| legal_principle | Dense | 4 | 0.1250 | 0.3750 | 0.1875 | 0.1499 |
| legal_principle | RRF | 4 | 0.2500 | 0.3750 | 0.3750 | 0.2272 |
| legal_principle | Reranker | 4 | 0.2500 | 0.5000 | 0.3750 | 0.2702 |
| procedural | BM25 | 7 | 0.2619 | 0.4762 | 0.4728 | 0.4026 |
| procedural | Dense | 7 | 0.1905 | 0.2381 | 0.4286 | 0.2875 |
| procedural | RRF | 7 | 0.2857 | 0.3571 | 0.4857 | 0.3598 |
| procedural | Reranker | 7 | 0.2381 | 0.4048 | 0.3321 | 0.3175 |
| semantic_paraphrase | BM25 | 5 | 0.3000 | 0.3000 | 0.2900 | 0.2577 |
| semantic_paraphrase | Dense | 5 | 0.3000 | 0.3000 | 0.1667 | 0.1897 |
| semantic_paraphrase | RRF | 5 | 0.3000 | 0.4000 | 0.3167 | 0.2956 |
| semantic_paraphrase | Reranker | 5 | 0.3000 | 0.4000 | 0.3500 | 0.3309 |
| statutory_interpretation | BM25 | 7 | 0.1905 | 0.2381 | 0.3952 | 0.2443 |
| statutory_interpretation | Dense | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| statutory_interpretation | RRF | 7 | 0.0000 | 0.1905 | 0.0799 | 0.0985 |
| statutory_interpretation | Reranker | 7 | 0.1429 | 0.1905 | 0.2619 | 0.1664 |

### Difficulty

| Category | System | N | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| easy | BM25 | 9 | 0.3426 | 0.3426 | 0.4815 | 0.3448 |
| easy | Dense | 9 | 0.2222 | 0.2778 | 0.2917 | 0.2387 |
| easy | RRF | 9 | 0.3056 | 0.3981 | 0.5103 | 0.3601 |
| easy | Reranker | 9 | 0.3704 | 0.4907 | 0.5481 | 0.4230 |
| hard | BM25 | 10 | 0.2000 | 0.2667 | 0.3493 | 0.2451 |
| hard | Dense | 10 | 0.1167 | 0.1167 | 0.1200 | 0.1117 |
| hard | RRF | 10 | 0.1167 | 0.1833 | 0.1992 | 0.1586 |
| hard | Reranker | 10 | 0.1833 | 0.2667 | 0.3611 | 0.2547 |
| medium | BM25 | 21 | 0.2857 | 0.3810 | 0.3468 | 0.3132 |
| medium | Dense | 21 | 0.1429 | 0.2143 | 0.2230 | 0.1787 |
| medium | RRF | 21 | 0.2222 | 0.3016 | 0.2901 | 0.2478 |
| medium | Reranker | 21 | 0.2143 | 0.3333 | 0.2813 | 0.2553 |

### Gold paragraph count

| Category | System | N | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| 1 | BM25 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | Dense | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | RRF | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 1 | Reranker | 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | BM25 | 22 | 0.3409 | 0.3864 | 0.3932 | 0.3312 |
| 2 | Dense | 22 | 0.2045 | 0.2500 | 0.2629 | 0.2115 |
| 2 | RRF | 22 | 0.2727 | 0.3409 | 0.3712 | 0.2891 |
| 2 | Reranker | 22 | 0.2727 | 0.3864 | 0.3782 | 0.3136 |
| 3 | BM25 | 16 | 0.2083 | 0.3125 | 0.3881 | 0.2924 |
| 3 | Dense | 16 | 0.1042 | 0.1667 | 0.1703 | 0.1478 |
| 3 | RRF | 16 | 0.1458 | 0.2500 | 0.2693 | 0.2199 |
| 3 | Reranker | 16 | 0.1667 | 0.2708 | 0.3146 | 0.2314 |
| 4 | BM25 | 1 | 0.2500 | 0.2500 | 0.2500 | 0.1681 |
| 4 | Dense | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 4 | RRF | 1 | 0.2500 | 0.2500 | 0.2000 | 0.1510 |
| 4 | Reranker | 1 | 0.0000 | 0.2500 | 0.1000 | 0.1128 |

### Judgment scope

| Category | System | N | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| multi_judgment | BM25 | 10 | 0.1833 | 0.3000 | 0.2476 | 0.2116 |
| multi_judgment | Dense | 10 | 0.1167 | 0.1167 | 0.1500 | 0.1200 |
| multi_judgment | RRF | 10 | 0.1167 | 0.2667 | 0.2050 | 0.1822 |
| multi_judgment | Reranker | 10 | 0.1500 | 0.2333 | 0.2333 | 0.1922 |
| single_judgment | BM25 | 30 | 0.3083 | 0.3583 | 0.4211 | 0.3339 |
| single_judgment | Dense | 30 | 0.1667 | 0.2333 | 0.2336 | 0.1940 |
| single_judgment | RRF | 30 | 0.2472 | 0.3028 | 0.3542 | 0.2736 |
| single_judgment | Reranker | 30 | 0.2722 | 0.3917 | 0.4040 | 0.3265 |

## Dense retrieval patterns

- `exact_terminology`: 6 queries; BM25 top-10 hits 5, dense hits 3; macro nDCG BM25=0.4378, dense=0.2619.
- `named_case_terms`: 4 queries; BM25 top-10 hits 4, dense hits 1; macro nDCG BM25=0.3304, dense=0.1968.
- `semantic_paraphrase`: 5 queries; BM25 top-10 hits 3, dense hits 2; macro nDCG BM25=0.2577, dense=0.1897.
- `statute_or_section_terms`: 23 queries; BM25 top-10 hits 16, dense hits 9; macro nDCG BM25=0.3038, dense=0.1616.
- The statutory-interpretation category is the clearest dense failure: 0/7 top-10 hit queries and macro nDCG@10 0.0000, versus BM25 macro nDCG@10 0.2443.
- Dense also trails BM25 on semantic-paraphrase queries, so the measured weakness is not confined to literal section or case terminology.
- Inspected failures frequently contain unlabelled neighboring provisions or nearby paragraphs from the correct judgment; exact paragraph-UID evaluation correctly gives these no gold credit.

## Query-level error sets

- BM25 succeeds / dense fails: 16 — Q005, Q006, Q007, Q008, Q011, Q013, Q014, Q017, Q019, Q024, Q029, Q031, Q033, Q036, Q038, Q040.
- Dense succeeds / BM25 fails: 2 — Q018, Q021.
- RRF hurts BM25 by nDCG: 14.
- RRF improves BM25 by nDCG: 9.
- Reranker improves RRF by nDCG: 13.
- Reranker hurts RRF by nDCG: 12.

Representative retrieved and gold text excerpts for these sets are stored in `retrieval_error_analysis.json`; conclusions here use those texts, not score magnitudes alone.

Across all 12 baseline reranker-worsened queries, the tracked text inspection shows promotions of advocacy/background, generic same-topic passages, and multiple nearby paragraphs from the same judgment above exact gold UIDs. Q027 and Q037 are direct examples. This is an ordering failure, not evidence for score blending.

## Paragraph length and splitting

- Corpus lengths: min 1, median 455.0, p95 2172, max 3601 characters.
- Very-short corpus paragraphs under 20 characters: 570 of 18822.
- Gold lengths: min 168, median 911, p95 2111, max 2551; under 20 characters: 0.
- Dense top-10 exact-label recall is 0.273 for 200–999-character gold paragraphs and 0.100 for 1,000+ characters; this is a correlation, not a demonstrated truncation cause.
- Very-short paragraphs appearing in top-10 results: BM25 0, dense 0, original RRF 0, original reranker 0. No gold label is under 20 characters.
- Stored paragraph text was not truncated or changed. Tokenizer-level truncation was not instrumented, so the experiment does not attribute long-paragraph misses to truncation.
- Canonical splitting was not changed because paragraph UIDs are the frozen relevance identity. Any boundary redesign requires a separately versioned corpus and gold set.

## Runtime and limitations

- Native top-50 cache: 86.977 s; cross-encoder score cache: 82.806 s; total experiment: 195.772 s.
- Metrics cover only 40 pilot queries and 100 Supreme Court judgments; no statistical significance is claimed.
- No embedding replacement, query rewriting, synonym expansion, score blending, BM25 sweep, or paragraph-boundary change was performed.
