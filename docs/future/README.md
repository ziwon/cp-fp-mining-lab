# Future Production Docs

This folder has one canonical future-state document:

- [`production_plan.md`](production_plan.md) — production architecture, hybrid
  deployment, DuckLake lineage, review sync, LLM/VLM pre-labeling, evaluation
  gates, and migration phases.

The goal is to keep future planning simple: current runnable behavior lives in
[`../current/`](../current/), stable reference material lives in
[`../reference/`](../reference/), and production roadmap material lives here.

## Status Boundary

The near-term production step is an optional DuckLake sync layer over the current
CSV/JSON outputs. That keeps the repo runnable offline while adding a durable
index for events, predictions, clusters, acquisition queues, human reviews,
dataset manifests, and gate results.
