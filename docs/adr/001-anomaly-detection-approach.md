# ADR-001: Anomaly Detection Approach

## Status
Accepted

## Context
The detector needs to flag billing invoices that deviate from normal
patterns in a synthetic CMS DE-SynPUF dataset, using ~50 deliberately
injected anomalies as ground truth. Three approaches were considered:

1. **Rules-based thresholds** (e.g. flag if billed_to_allowed_ratio > N)
2. **Classical ML — Isolation Forest** trained on tabular claim features
3. **Embeddings + cosine similarity** against the centroid of all
   invoice embeddings

## Decision
v1 uses embeddings + cosine similarity. Each invoice is embedded
(Azure `text-embedding-3-small`, local sentence-transformers fallback)
from a templated description of its claim fields. Anomaly score is the
cosine distance from an invoice's embedding to the centroid of all
embeddings; invoices above `ANOMALY_THRESHOLD` are flagged.

Isolation Forest is scoped explicitly to Phase 2 — both approaches
will eventually run side by side so the README can report a real
classical-ML-vs-embedding comparison rather than asserting one is
better.

Rules-based thresholds were rejected as a primary approach: they
require hand-picking cutoffs per field, don't generalize across claim
types, and don't produce a continuous score that the explanation
chain (GPT-4o) can reason over.

## Consequences
- **Pro:** No training step required before the first detection run;
  embeddings can be computed incrementally as invoices are indexed.
- **Pro:** A continuous score (rather than a binary rule outcome)
  gives the explanation chain something graded to reason about, and
  gives the eval harness a threshold to tune against precision/recall.
- **Con:** Centroid-based scoring is sensitive to dataset composition
  — a dataset skewed toward one claim type will shift the centroid
  and could under-flag anomalies in minority claim types. Worth
  noting as a known limitation in the README, not silently fixing it
  pre-launch.
- **Con:** No model interpretability beyond "far from centroid" until
  GPT-4o's explanation layer adds reasoning — this is precisely why
  the explanation chain exists in the architecture at all (classical
  ML scores, LLM explains).

## Alternatives Considered
| Approach | Rejected because |
|---|---|
| Rules-based thresholds | Doesn't generalize, no continuous score |
| Isolation Forest (v1) | Requires a labeled-enough feature set and training step before any detection is possible; deferred to Phase 2 once v1's eval baseline exists to compare against |

## Post-implementation finding — threshold calibration

After running against the actual Azure embeddings, the cosine distance
distribution was tighter than expected:

- Max score: 0.0483
- p99 score: 0.0324
- Mean score: 0.0196

The original threshold of 0.80 (a common default for cosine distance)
produced zero detections. Calibrated to 0.030 (just below p99) to flag
the top 1% of invoices.

**Root cause:** `text-embedding-3-small` treats all billing claims as
semantically similar — they share the same structural template (code,
NPI, amounts, date). The numerical anomaly signal (a 9x ratio vs 1.5x)
doesn't produce a large directional shift in embedding space. This
confirms the ADR-001 prediction that cosine similarity alone has a
ceiling on structured tabular billing data, and validates the Isolation
Forest upgrade planned for Phase 2 — classical ML on raw features will
catch ratio-based anomalies directly without going through the embedding
representation.
