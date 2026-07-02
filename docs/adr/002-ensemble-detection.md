# ADR-002: Ensemble Detection — Isolation Forest + Cosine Similarity
 
## Status
Accepted — 2026-07
 
## Context
ADR-001 documented that cosine similarity over text-embedding-3-small
embeddings is insufficient as a standalone anomaly detector for structured
billing claims. Measured on the 5,050-claim DE-SynPUF dataset with 50
injected anomalies:
 
- Max cosine distance from centroid: 0.0483
- Mean cosine distance: 0.0196
- v1 results (threshold = p99 = 0.030): Precision 0.079, Recall 0.200, F1 0.114
 
The embedding model compresses structurally different claims into nearly
identical vectors. Distance thresholds cannot separate what the embedding
space does not distinguish. We need a detector that operates on explicit
domain features rather than a scalar distance.
 
## Decision
1. Introduce a domain port, AnomalyDetectorPort, with two infrastructure
   adapters: CosineSimilarityDetector (existing logic, refactored) and
   IsolationForestDetector (new).
2. IsolationForestDetector consumes a structured feature vector built from
   billing domain signals (claim totals, line statistics, provider claim
   frequency, procedure diversity, and centroid cosine distance) — NOT the
   raw 1,536-dimension embedding.
3. Combine detector outputs in an application-layer
   EnsembleDetectionService using a weighted average:
   final = 0.4 * cosine_score + 0.6 * isolation_forest_score.
   Both scores are normalized to [0, 1] by their adapters (normalization
   is part of the port contract).
4. The Isolation Forest is trained offline and persisted with joblib.
   The API never trains; it only scores.
 
## Alternatives Considered
- One-Class SVM: strong on small datasets but O(n^2)-ish training,
  sensitive to kernel/nu tuning, and less interpretable. Rejected.
- DBSCAN: density clustering, not a scoring model — produces labels, not
  ranked anomaly scores, and has no natural predict() for unseen claims.
  Rejected.
- Autoencoder reconstruction error: requires a training loop, GPU-adjacent
  tooling, and far more tuning for a 5,050-row dataset. Overkill. Rejected.
- Isolation Forest: linear-time training, native anomaly scoring via
  score_samples(), a contamination parameter that maps directly to our
  known ~1% injection rate, works well on low-dimensional tabular
  features, and ships in scikit-learn. Selected.
 
## Feature Representation Decision
Raw embeddings (Option A) were rejected as IF input: 1,536 dimensions of
low-variance signal is exactly what failed in v1. Structured features
(Option B) selected: 8 domain-driven features (see features.py). This puts
healthcare billing domain knowledge — the portfolio differentiator —
directly into the model input.
 
## Ensemble Weighting Rationale
0.6 IF / 0.4 cosine: IF sees the full feature space and is expected to
carry more signal; cosine is retained because embedding distance remains a
legitimate (weak) independent signal and keeping it demonstrates the
ensemble pattern. Weights live in Settings, not code, and are a documented
tuning surface.
 
## Threshold Calibration Note
The ensemble threshold is calibrated against the labeled injected
anomalies (maximizing F1). This is label leakage in a strict ML sense and
is acceptable here because (a) the labels are synthetic, (b) the project's
purpose is architectural demonstration, and (c) the leakage is disclosed —
in production, thresholds would be calibrated on a held-out labeled set or
via analyst feedback loops.
 
## Consequences
- Two new runtime dependencies: scikit-learn >= 1.4, joblib >= 1.3.
- A trained artifact (models/isolation_forest.joblib) becomes a deployment
  concern: versioned alongside code, loaded at API startup.
- Feature extraction logic becomes a correctness-critical, unit-tested
  module — feature drift between training and serving silently breaks
  scoring (train/serve skew).
- v2 evaluation results: 
  - Cosine-only:   P=____  R=____  F1=____
  - IF-only:       P=____  R=____  F1=____
  - Ensemble:      P=____  R=____  F1=____
- Revisit trigger: if ensemble F1 < 0.5 on the injected set, escalate to
  supervised classification (labels exist) or revisit feature engineering.
