# Docs

The docs are organized by repo status:

- [`current/`](current/) — implemented behavior that can run in this repository
  today.
- [`future/`](future/) — production architecture and roadmap material.
- [`reference/`](reference/) — stable taxonomy/reference material.
- [`assets/`](assets/) — screenshots and architecture graphics used by the docs.

## Current Implementation

- Local synthetic demo: [`current/demo_walkthrough.md`](current/demo_walkthrough.md)
- Real D-Fire/YOLO pipeline: [`current/real_data.md`](current/real_data.md)
- Label Studio setup/import/export:
  [`current/label_studio_setup.md`](current/label_studio_setup.md)
- Compact lab architecture: [`current/architecture.md`](current/architecture.md)
- Active-learning loop: [`current/active_learning.md`](current/active_learning.md)

## Future Production Design

- Production plan:
  [`future/production_plan.md`](future/production_plan.md)

## Reference

- False-positive label taxonomy:
  [`reference/fp_taxonomy.md`](reference/fp_taxonomy.md)

## Suggested Reading Paths

- First-time local demo:
  [`current/demo_walkthrough.md`](current/demo_walkthrough.md) ->
  [`current/label_studio_setup.md`](current/label_studio_setup.md)
- Real-data experiment:
  [`current/real_data.md`](current/real_data.md) ->
  [`reference/fp_taxonomy.md`](reference/fp_taxonomy.md) ->
  [`current/label_studio_setup.md`](current/label_studio_setup.md)
- Production migration:
  [`current/architecture.md`](current/architecture.md) ->
  [`current/active_learning.md`](current/active_learning.md) ->
  [`future/production_plan.md`](future/production_plan.md)
