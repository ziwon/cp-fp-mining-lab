# Demo Walkthrough — Full Local Active-Learning Loop

End-to-end run of the false-positive mining loop on the local Docker stack:
synthetic data → uncertainty-ranked review tasks → human labeling in Label Studio
→ webhook retrain → eval gate → model promotion → ML backend serves it → W&B
lineage. Everything runs locally; no cloud accounts required.

For the design see [`active_learning.md`](active_learning.md); for Label Studio
specifics see [`label_studio_setup.md`](label_studio_setup.md).

## Services and ports

The stack remaps host ports because the defaults (8080/9000) are often taken; the
values below come from `.env` (copy from `.env.example`). Container-internal ports
are unchanged.

| Service | Browser URL (host) | In-network address | Role |
| --- | --- | --- | --- |
| Label Studio | http://localhost:18081 | — | review UI |
| ML backend | http://localhost:9090 | `http://ml-backend.local:9090` | serves detector predictions (Phase 2) |
| Webhook | http://localhost:9091 | `http://webhook.local:9091` | retrain trigger (Phase 3) |
| Fileserver | http://localhost:18090 | — | serves task images to the browser |
| W&B Server | http://localhost:18082 | `http://wandb:8080` | runs, Tables, model artifacts |
| MinIO | http://localhost:19001 | `http://minio:9000` | object store (wired, not yet in data path) |

Inside Label Studio's settings, always use the **`.local` in-network addresses**
(`ml-backend.local`, `webhook.local`) — LS reaches them over the Docker network,
and its URL validator rejects bare hostnames without a dot.

## 0. Bring the stack up

```bash
cp .env.example .env          # then edit ports/creds if needed
just docker-build             # build the miner/services image
docker compose up -d minio labelstudio wandb fileserver ml-backend webhook
```

The `miner` container is one-shot (exits immediately) — that is expected; the six
services above are the long-running ones.

## 1. Generate data, cluster, rank, train

```bash
# full batch pipeline (00 -> 06): data, embeddings, clusters, ranked tasks,
# sample review import, W&B table, and the bootstrapped detector
just docker-demo
```

Or run the active-learning selection alone (top-N most informative tasks):

```bash
docker compose run --rm miner python scripts/03_export_for_label_studio.py \
  --budget 30 --strategy entropy
```

This writes `data/processed/labelstudio_tasks.json` (task images as
`http://localhost:18090/...` URLs) and bootstraps `model_registry/` with a
production detector.

## 2. Configure Label Studio (two separate steps)

These are different artifacts going to different places — mixing them up is the
most common mistake (see [`label_studio_setup.md`](label_studio_setup.md)).

1. **Labeling interface** — `Settings → Labeling Interface → Code`, paste
   `data/processed/labelstudio_label_config.xml`, Save. (If this is empty,
   clicking a task redirects to "Go to setup".)
2. **Tasks** — `Import` → `data/processed/labelstudio_tasks.json`.

Connect the live model loop:

- `Settings → Model → Connect Model` → `http://ml-backend.local:9090`
- `Settings → Webhooks → Add` → `http://webhook.local:9091/webhook`, **Send
  payload ON**, trigger on *Annotation created*.

## 3. Label tasks (drives the loop)

Click **Label All Tasks**. For each task you **must** select **is_event**
(required) and **fp_type** — `fp_type` is the field the webhook learns from. A
blank submit creates an empty annotation that teaches nothing.

After every `webhook.batch_threshold` annotations the webhook:

1. persists labels → `data/processed/webhook_reviews.csv`
2. retrains the detector on bootstrap labels overridden by your corrections
3. evaluates on a hold-out
4. gate-promotes (or rejects) the candidate
5. logs a W&B `retrain` run with metrics, the model artifact, and a
   `review_batch` Table of the exact labels that drove it

Watch it live:

```bash
docker compose logs -f webhook
```

## 4. The promotion gate

`detector.promotion_min_delta` controls promotion
(`candidate >= production + delta`):

- `0.0` — **strict** (default): promote only if the candidate at least matches
  production. Protects production from regressions.
- `> 0` — require a real improvement of that size.
- `< 0` — tolerance: accept a candidate within `|delta|` of production.

When a candidate is promoted, the registry `production` alias moves and the ML
backend (which loads `production` per request) serves the new model immediately —
verify with `curl localhost:9090/health`.

> Note: on the synthetic dataset the bootstrap labels are already near-perfect, so
> human corrections rarely *beat* production under the strict gate — expect most
> review batches to be rejected (gate working as intended). Use a tolerance gate
> (`< 0`) only to demonstrate the promote→serve swap.

## 5. W&B lineage

http://localhost:18082/aaron/cv-fp-mining-lab — filter runs by `job_type`:

- `retrain` — one run per review batch: `macro_f1`/`accuracy`/`promoted`, the
  model artifact (alias `production` or `candidate`), and the `review_batch` Table.
- `fp-curation` — from `scripts/05_log_to_wandb.py`: the reviewed-samples Table +
  dataset artifact (a manual pipeline step, not triggered by labeling).

W&B is **offline by default**; to log to the self-hosted server, create a user in
the W&B UI, put its key in `.env` (`WANDB_API_KEY`), set `WANDB_MODE=online`, and
recreate the affected services.

## Troubleshooting (issues hit during this demo)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Webhook URL "Enter a valid URL." | LS rejects bare hostnames | use `http://webhook.local:9091/webhook` |
| Clicking a task → "Go to setup" | labeling interface not set | paste the label config XML in project settings |
| "No More Tasks Left in Queue" | tasks already annotated (often empty) | delete empty annotations to re-queue |
| Broken image icons | image port not reachable / no CORS header | tunnel/expose `FILESERVER_PORT`; fileserver sends CORS (`docker/fileserver.conf`) |
| One/two images broken, rest fine | browser cached a pre-CORS response | empty-cache hard reload; server sends `Vary: Origin` to prevent recurrence |
| W&B run "offline" | `WANDB_MODE=offline` | set API key + `WANDB_MODE=online`, recreate services |

## Teardown

```bash
docker compose down            # stop services (volumes persist)
just clean                     # remove generated data/ artifacts
```
