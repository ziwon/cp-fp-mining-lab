# Real-Data Track — D-Fire detector + false-positive mining

The default pipeline mines **synthetic** false positives. This track replaces the
synthetic generator (`scripts/00`) with a real loop on the
[D-Fire](https://github.com/gaiasd/DFireDataset) fire/smoke detection dataset:
train a real detector, run it, and harvest its **actual** false positives as the
mining input. Everything downstream (embeddings → cluster → review → W&B) consumes
the same `fp_events.csv` schema unchanged.

This is steps 1–2 of the production-hardening plan in
[`active_learning.md`](active_learning.md): a real detector and real FP collection.

## Install

```bash
just setup-real                 # uv sync --extra detect --extra clip
```

CPU works for small subsets. For GPU, set `yolo.device: 0` (and
`embedding.device: cuda`) in `configs/pipeline.yaml`.

### GPU setup (NVIDIA Blackwell, RTX 50xx / sm_120)

Blackwell needs a recent CUDA torch build (the default PyPI torch fails with a
sm_120 kernel mismatch). With driver 580+ (CUDA 13 capable), a dedicated GPU venv:

```bash
uv venv .venv-gpu --python 3.10
uv pip install --python .venv-gpu/bin/python --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch torchvision -e . ultralytics "transformers>=4.40,<5"
# verify
.venv-gpu/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Verified on an RTX 5080: torch 2.12.1+cu130, `cuda available: True`, capability
(12, 0); CLIP embeds ~35 img/s and YOLO predicts on `device=0`. On this host the
GPU venv lives at `/data/venvs/cpfp-gpu` (small workspace disk).

## Configure

The `dfire`, `yolo`, and `mining` blocks in `configs/pipeline.yaml` point at the
dataset and output locations. Defaults use the 300-image `dfire-300-stratified`
subset for fast iteration; switch `dfire.train_dir` to the full `D-Fire/train` for
real training. Heavy outputs (runs, weights, crops) default to `/data/...` because
the workspace disk is small on this host.

## Run

```bash
just detect-prepare     # write the Ultralytics dataset YAML (0=smoke, 1=fire)
just detect-train       # fine-tune yolov8n on D-Fire -> best.pt
just mine-fp            # run the detector over a split, harvest FP crops
# then the normal pipeline on the real FPs:
uv run python scripts/01_extract_embeddings.py --method simple
uv run python scripts/02_cluster_false_positives.py
uv run python scripts/03_export_for_label_studio.py
```

`mine-fp` writes `data/processed/fp_events.csv` (same schema as the synthetic
generator, plus `source_image_path` and `is_negative_image`) and the FP crops.

For production-shaped active learning, treat the synthetic pipeline as a smoke test
and run Label Studio/W&B on this real-data track:

```bash
# Mine + cluster real FP crops.
just mine-fp
uv run python scripts/01_extract_embeddings.py --method clip
uv run python scripts/02_cluster_false_positives.py
uv run python scripts/03_export_for_label_studio.py --budget 200 --strategy entropy

# Import reviewed real FP crops, then build hard negatives from the reviewed export.
uv run python scripts/04_import_label_studio_export.py --input <label-studio-export.json>
just build-hardneg
just retrain-hardneg
uv run python scripts/13_evaluate_and_gate.py --candidate /data/cpfp-output/yolo_runs/dfire-hardneg/weights/best.pt
```

`scripts/03_export_for_label_studio.py` preserves `source_image_path`, so reviewed
exports can feed `scripts/14_build_hard_negatives.py` directly. Script 14 now prefers
`data/processed/reviewed_fp_samples.csv` when present, falling back to `fp_events.csv`
for fully headless GT-confirmed runs.

## First LS + W&B workflow smoke test

After the headless real-data pipeline works, run one small Label Studio + W&B pass
to validate the production workflow wiring. The goal of this pass is **lineage and
handoff verification**, not model improvement; 30-50 reviewed crops are enough.

1. Install the real-data extras and start the local review/lineage services:

   ```bash
   just setup-real
   just docker-build
   docker compose up -d minio labelstudio wandb fileserver
   ```

   For online W&B logging, create a user/API key in the W&B UI, set
   `WANDB_MODE=online`, `WANDB_BASE_URL`, and `WANDB_API_KEY`, then recreate the
   worker shell/service. Without those values the scripts log offline.

2. Mine and rank a small real FP review batch:

   ```bash
   WANDB_MODE=online just mine-fp
   uv run python scripts/01_extract_embeddings.py --method clip
   uv run python scripts/02_cluster_false_positives.py
   uv run python scripts/03_export_for_label_studio.py --budget 50 --strategy entropy
   ```

3. In Label Studio:

   - Set the labeling interface from `data/processed/labelstudio_label_config.xml`.
   - Import `data/processed/labelstudio_tasks.json`.
   - Review 30-50 tasks, selecting `is_event`, `fp_type`, and `bbox_valid`.
   - Export the reviewed JSON.

4. Build, retrain, and gate from the reviewed export:

   ```bash
   uv run python scripts/04_import_label_studio_export.py --input <label-studio-export.json>
   WANDB_MODE=online just build-hardneg
   WANDB_MODE=online just retrain-hardneg
   WANDB_MODE=online uv run python scripts/13_evaluate_and_gate.py \
     --candidate /data/cpfp-output/yolo_runs/dfire-hardneg/weights/best.pt
   ```

5. Verify the pass:

   - Label Studio tasks render FP crops and reviewer choices are present in the export.
   - `data/processed/reviewed_fp_samples.csv` keeps `source_image_path`.
   - `scripts/14_build_hard_negatives.py` uses the reviewed CSV, not stale headless
     `fp_events.csv`.
   - W&B has `mine-fp`, `build-hardneg`, `yolo-retrain`, and `eval-gate` runs with
     linked artifacts.
   - The gate either promotes or rejects the candidate with explicit check results.

If this smoke test passes, tune for quality with a larger, more diverse FP batch,
oversampling, lower fine-tuning LR, and more epochs.

## How a false positive is defined

A detection is a false positive when it matches **no ground-truth box** — same
class and IoU ≥ `mining.iou_thr` is a match (`fp_mining.is_false_positive`). Two
sources dominate:

- **Negative images** (empty YOLO label = no fire/smoke): any detection is an FP.
  D-Fire's train split has ~7,800 such images — the richest FP source.
- **Misclassification / mislocalization** on positive images: e.g. predicting
  `fire` over a `smoke` region, or a box that misses the object.

The crop of each FP region becomes a review task — the real analogue of the
synthetic steam/reflection/headlight samples.

## Trained detector

A yolov8n trained on full D-Fire (17,221 train / 4,306 val, 50 epochs) on an
RTX 5080 reaches **P 0.77, R 0.70, mAP@50 0.77, mAP@50-95 0.45** on the test set —
in line with published D-Fire results. With this detector, mining at the normal
`conf=0.25` yields realistic confusions (mislocalized boxes + clouds/lights),
unlike the underfit smoke-test model that only fired at `conf≈0.01`.

## Notes & next steps

- An **underfit** detector (few epochs) yields *more* FPs — useful for exercising
  the loop, but use a properly trained detector and a sane `mining.conf` for real
  curation (the config defaults to full D-Fire + 50 epochs).
- `simple` embeddings (color/edge histograms) cannot separate real FP types well;
  clusters collapse. **Step 3 (done)** implements the `clip` path
  (`uv sync --extra clip`, then `--method clip` on script 01): on the 244 mined
  D-Fire FPs, CLIP took HDBSCAN from 2 collapsed clusters to 9, with fire FPs
  separating into their own clusters. (Euclidean silhouette undersells CLIP since
  it lives in cosine space — judge by cluster structure, not euclidean distance.)
## Detection-aware promotion gate (step 4)

`scripts/13_evaluate_and_gate.py` (`just eval-gate`) replaces the single-scalar
hold-out gate with a multi-criteria check on the frozen eval set
(`detection_eval` + `gating`). A candidate is promoted only if it clears **all**:

- **mAP@50** doesn't regress (`gate.map50_min_delta`),
- **per-class recall** stays above the floor vs production (`gate.recall_min_delta`)
  — keeps detecting real fire *and* smoke,
- **negative-image FP-rate** doesn't increase (`gate.fp_rate_max_delta`) — the
  cost FP mining is meant to drive down.

Why multi-criteria matters: an underfit model that detects *nothing* has a perfect
FP-rate (zero false positives) and would sail through an FP-only gate. Verified on
real models — the underfit 300-image detector (mAP 0.008, recall ~0,
neg_fp_rate 0.000) was **rejected on mAP and per-class recall** against the full
detector (mAP 0.69, recall 0.75, neg_fp_rate 0.016), so production was protected.

Promoted detectors are stored in `gate.registry_dir` (`LocalModelRegistry.register_file`,
`.pt` artifacts with `candidate/staging/production` aliases).

Gate runs are also logged to W&B (`job_type=eval-gate`) with candidate/production
metrics, per-check pass/fail detail, the registered YOLO `.pt` artifact, and the
current hard-negative dataset artifact when present. W&B remains offline by default;
set `WANDB_MODE=online`, `WANDB_BASE_URL`, and `WANDB_API_KEY` to sync to the local
or hosted server.

The upstream real-data stages also emit W&B lineage:

- `mine-fp`: `fp_events.csv`, FP crop artifact, mining source paths, threshold config.
- `build-hardneg`: reviewed/confirmed source CSV, YOLO hard-negative dataset artifact,
  image/label counts, empty-label count.
- `yolo-retrain`: warm-start weights, combined dataset YAML, hard-negative counts,
  candidate `.pt` artifact.
- `eval-gate`: candidate vs production metrics, gate checks, promotion decision,
  registered model artifact.

## Closing the loop — hard-negative retraining

The real detector loop is closed end-to-end:

```bash
just mine-fp        # (12) confirmed FPs from a non-eval pool (train/external)
just build-hardneg  # (14) FP source frames -> validated YOLO hard-negative dataset
just retrain-hardneg# (15) fine-tune production on [base train + hard negatives]
just eval-gate -- --candidate <new best.pt>   # (13) gate decides promotion
```

`dataset_builder.build_hard_negative_dataset` copies each confirmed FP's source
frame with its **validated** ground-truth label (empty for background images), so
the detector relearns the correct content of FP-prone frames. Invalid/out-of-bounds
labels are dropped (the dataset-validation guard).

**Honest result on this run:** mining 58 hard negatives from D-Fire train and
fine-tuning 3 epochs *did not* improve the detector — neg-FP-rate rose
(0.016 → 0.024) and mAP/recall dipped, so the gate **rejected** the candidate and
kept production. That is the loop working: a naive retrain (too few hard negatives,
full-schedule LR restart over a converged model) is exactly what the gate exists to
stop. Making it actually improve needs many more hard negatives (or heavy
oversampling), a low fine-tuning LR, and more epochs — a tuning exercise on top of
the now-working mechanism.

## Remaining

- Add the `verifier-eval` video clips to the gate (temporal event-level FP-rate).
- Tune the hard-negative retrain (LR/volume/oversampling) so candidates clear the
  gate.
- Replace the local registry with W&B Registry aliases or another production model
  registry while keeping the same `candidate → staging → production` gate contract.
