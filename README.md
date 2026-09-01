# Frozen DINOv3 vs V-JEPA 2.1 for COD

The project covers Milestones A–I: project/data foundations, frozen backbone
adapters, the common decoder/loss, reproducible training and evaluation, Colab
workflows, and the full frozen-backbone comparison with qualitative overlays.

## Setup

Use Python 3.11, clone the official `facebookresearch/dinov3` and
`facebookresearch/vjepa2` repositories outside this repository, acquire checkpoints
through their official procedures, copy `.env.example` values into your environment,
then install and test:

```bash
python -m pip install -e '.[dev]'
pytest
python scripts/runtime_check.py
python scripts/inspect_backbone.py --backbone dinov3_vitb16 --image /path/to/cod.jpg
python scripts/inspect_backbone.py --backbone vjepa21_vitb16 --image /path/to/cod.jpg
```

Both commands enforce four `[B,768,24,24]` frozen feature maps from layers
`[2,5,8,11]`. DINOv3 uses normalized reshaped patch tokens. V-JEPA 2.1 loads only
the checkpoint's `ema_encoder` state and sends `[B,C,1,H,W]` through its native
`img_temporal_dim_size=1` image pathway; it does not repeat images and never uses
the predictor. Unexpected token layouts fail rather than being guessed.

The smoke report includes upstream commit/checkpoint hashes, parameter counts,
runtime, dtype, feature shapes, peak CUDA memory, and mean forward time.

## Data

Manifests contain `id,source,image_path,mask_path`. Generate individual manifests
with `scripts/prepare_manifests.py`, then validate the locked protocol with:

```bash
python scripts/validate_dataset.py --manifest-dir manifests
```

Create the deterministic, source-stratified 90/10 development split only from the
validated training manifest with `python scripts/prepare_manifests.py
--split-train manifests/train_all.csv`.

The seven COD10K duplicate pairs listed in `configs/dataset_exclusions.csv` are
documented or SHA-256-confirmed overlaps between the official train and test splits.
Manifest bootstrapping removes each side's corresponding ID, producing 4,033 training
and 2,019 COD10K-Test rows. The
validator requires those effective counts, rejects excluded IDs and duplicate paths,
opens every pair, and hashes images to require zero remaining train/test overlap.
Notebook 03 stores a versioned receipt in Drive containing manifest hashes, effective
counts, settings, and completion time; an exact match skips repeat full validation.

## Colab workflow

The local Git repository remains the source of truth: edit/commit/push locally,
then pull and reinstall in Colab. A minimal bootstrap is:

```bash
git clone YOUR_REPOSITORY_URL /content/cod-ssl
git clone https://github.com/facebookresearch/dinov3 /content/third_party/dinov3
git clone https://github.com/facebookresearch/vjepa2 /content/third_party/vjepa2
pip install -e '/content/cod-ssl[dev,notebooks]'
```

Mount Drive if desired, export the four paths from `.env.example`, and run the two
inspection commands above. Credentials, access URLs, datasets, weights, and runs
must remain outside Git.

## Training

After preparing and validating `manifests/train_all.csv`, run the required small
smoke experiment before a full run:

```bash
python scripts/train.py --config configs/frozen_dinov3_vitb16.yaml --limit-train 32 --epochs 2
python scripts/train.py --config configs/frozen_vjepa21_vitb16.yaml --limit-train 32 --epochs 2
```

The optimizer contains decoder parameters only. Runs record resolved configuration,
environment/upstream versions, CSV and TensorBoard logs, and resumable decoder-only
checkpoints. Use `--resume runs/.../checkpoints/last.pt --run-dir runs/...` to resume.

Evaluate a completed run on all four locked test sets with:

```bash
python scripts/evaluate.py --run runs/<run-directory>
```

The evaluator resizes 384×384 logits to each original GT size, applies sigmoid and
per-image min–max normalization, saves the exact uint8 PNG consumed by
`pysodmetrics`, and writes `metrics.json`/`metrics.csv`. Test manifests are never
used by the training CLI.

Every Colab notebook is independently runnable from a fresh GPU kernel. Its first
cell mounts Drive, clones or fast-forwards the project, installs it, and invokes the
shared idempotent `scripts/bootstrap_colab.py`. That script reuses cached Drive
weights/data, prepares missing prerequisites, restores environment paths, and asks
for the private DINOv3 URL only when its checkpoint is absent. There is no required
notebook execution order.

For the controlled GPU smoke gate, run `notebooks/02_frozen_baseline_smoke_train.ipynb`.
The shared bootstrap downloads the official combined training archive,
caches/extracts it in Drive, discovers an unambiguous 4,040-pair layout, and builds
the manifest after the user explicitly acknowledges the COD10K non-commercial
license. Notebook 02 first writes a seed-42, source-proportional 256-row manifest
(63 CAMO and 193 COD10K images) from the decontaminated 4,033-image pool, distributing the COD10K allocation across every
filename-derived category. Both backbones train from that exact persisted manifest
for five epochs. Each run snapshots and hashes its training manifest, exports a
sample prediction, and reloads each checkpoint to verify finite logits and freezing.
Its final cells also evaluate both checkpoints on all 256 smoke-training images,
export per-image Dice/IoU/MAE/uncertainty values, aggregate COD metrics, descriptive
and paired statistics (bootstrap confidence intervals, Wilcoxon tests, effect sizes,
and win rates), CSV/Markdown/LaTeX tables, diagnostic plots, and six score-selected
qualitative panels. These artifacts are explicitly training-subset diagnostics, not
held-out publication evidence.

After both intermediate smoke runs pass, independently run
`notebooks/03_full_frozen_comparison.ipynb`. It downloads/caches the four test sets,
validates the locked counts and train/test isolation, runs the two full 40-epoch
experiments, evaluates each dataset separately, and writes its results under Drive.
Stable run names and epoch checkpoints allow a disconnected Colab session to resume.
The comparison directory contains metric and compute CSVs, training/metric graphs,
and 24 paired panels showing the original, ground-truth boundary, probability masks,
and red prediction overlays for both backbones.

Use `notebooks/04_all_layer_mixture.ipynb` for the DINOv3 follow-up ablation.
DINOv3 exposes all 12 frozen transformer layers; 48 trainable scalar logits form four
softmax-weighted mixtures for the unchanged common decoder. The notebook reuses the
same 40-epoch, 4,033-image protocol, exports the learned 4×12 matrix and heatmap, and
compares it with fixed-layer DINOv3. It also writes a new extended table and graph
containing notebook 03's fixed DINOv3 and official-layer V-JEPA results plus the new
DINOv3 all-layer result; notebook 03 and its artifacts are never overwritten.
