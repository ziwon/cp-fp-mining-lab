# Current Implementation Docs

These docs describe what the repository can run today. Start here when setting
up the lab, running the synthetic demo, using the real D-Fire/YOLO path, or
debugging the current Label Studio and active-learning loop.

## Recommended Order

1. [`architecture.md`](architecture.md) — compact map of the implemented lab.
2. [`demo_walkthrough.md`](demo_walkthrough.md) — local synthetic active-learning
   demo.
3. [`label_studio_setup.md`](label_studio_setup.md) — Label Studio import,
   interface setup, image rendering, and production review-sync notes.
4. [`active_learning.md`](active_learning.md) — implemented active-learning
   pieces and hardening still left.
5. [`real_data.md`](real_data.md) — D-Fire/YOLO real-data mining and retraining
   path.

## Status Boundary

The current implementation is intentionally lightweight: local files, Docker
Compose, optional W&B, and offline fallbacks. Future production concerns such as
Argo Workflows, DuckLake-backed lineage, Kubernetes namespace design, HA
PostgreSQL, and governance controls live under [`../future/`](../future/).
