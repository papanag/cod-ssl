# Frozen DINOv3 vs V-JEPA 2.1 for COD

The project covers Milestones A–I: project/data foundations, frozen backbone
adapters, the common decoder/loss, reproducible training and evaluation, Colab
workflows, and the full frozen-backbone comparison with qualitative overlays.

It also contains the gated implementation for the focused VCOD extension: DS
(DINO image), VI (official V-JEPA2.1 image path), DT (framewise DINO plus
GatedMambaMix), and VV (native V-JEPA2.1 video). DM and VR are diagnostics only.

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

For DT training, install the optional CUDA GMMix dependency with
`pip install -e '.[dev,vcod]'`. The reference provenance and pinned commit are
recorded in `docs/gmmix_provenance.md`.

## Focused video study

Video data is represented by a canonical CSV manifest with these required
columns: `dataset,regime,split,video_id,source_video_id,frame_id,frame_number,
image_path,mask_path,annotation_type`. Optional `fps` and JSON `attributes`
columns are preserved. Corrected manifests also store `sequence_position`,
`source_frame_number`, release/cadence fields, and boundary policy. The optional Boolean `is_target` column permits a row to
provide temporal context without claiming that it has a supervised mask. Paths
are absolute so run snapshots remain unambiguous.

Notebook 05 runs `scripts/bootstrap_vcod_data.py`: it downloads and resumes the
official Original MoCA, MoCA-Mask, and CAMotion archives, caches selected release assets in Drive,
verifies locked release counts, creates deterministic video-disjoint validation
splits from official training sequences, and writes manifests, split IDs, archive
hashes, and provenance receipts. MoCA uses only 4,691 manual targets; dense context
comes from SHA-256-verified Original MoCA frames within conservative target hulls.
No pseudo masks enter the corrected benchmark. CAMotion uses only its 30,028 sequence-organized
manual RGB/GT pairs and the attribute file pinned to official repository commit
`bf92692f9f9f2820185f9aa9a06fd2891dadf9a7`. The public archive does not contain
the paper-described 149,319 collected frames; released observations have source-frame step five.

For a non-Colab bootstrap, run:

```bash
python scripts/bootstrap_vcod_data.py \
  --data-root /persistent/vcod/data --manifest-dir /persistent/vcod/manifests \
  --datasets moca_mask_dense camotion --accept-camotion-academic-license
```

Dataset releases, checkpoints, and outputs stay outside Git. Execute the
correctness gates before training:

```bash
python scripts/inspect_dataset.py --config configs/datasets/moca_mask_dense.yaml \
  --output outputs/inspection/moca_mask_dense
python scripts/inspect_dataset.py --config configs/datasets/camotion.yaml \
  --output outputs/inspection/camotion
python scripts/inspect_backbone.py --model vjepa21_vitb16 --pathway video \
  --clip-length 64 --target-index 32
pytest
```

The official V-JEPA2.1 hub geometry locks the primary clip to 64 observations,
tubelet size 2, and source target index 32. MoCA primary context uses source stride
one; CAMotion uses released stride one/source stride five. The selected temporal
token is 16 and covers source indices 32–33. See
`docs/vjepa21_dense_mapping.md`; manual checkpoint-specific sign-off is still
mandatory before a scientific test run.

The Colab workflow is split into three thin, independently restartable
notebooks:

- `notebooks/05_vcod_setup_validation.ipynb` runs dataset inspection,
  cross-dataset/source-pairing audits, backbone mapping checks, visual QA, and
  writes the manual approval receipt.
- `notebooks/06_vcod_run_cell.ipynb` runs one smoke, validation-tuning, final,
  or diagnostic cell at a time with a stable Drive path and automatic resume.
- `notebooks/07_vcod_summarize.ipynb` reads only `vcod/runs`, requires every
  declared primary cell, and generates the paired report.

Keep canonical manifests under `MyDrive/cod-ssl/vcod/manifests`, and never copy
smoke or tuning outputs into the final `vcod/runs` directory. The VCOD trainer
writes readout-only atomic checkpoints every 250 optimizer steps so a Colab
disconnect loses bounded work and cannot leave a partially written checkpoint.
Notebook 06 tunes with successive halving at 250, 1,000, and 3,000 optimizer
steps. All three rates run at stage one, the best two per dataset/system advance,
and only the best advances to stage three. Stage validation artifacts and immutable
promotion receipts are retained under `vcod/tuning`; final runs always start clean.
Its separate `exploratory` queue runs MoCA-D1 for DS/VI/DT/VV with 75 optimizer
steps, 512 training targets, and 256 validation targets. Both subsets use the
same deterministic, seeded round-robin sampling across source videos for every
system. These quick-look artifacts stay under `vcod/exploratory`, are marked
non-primary, and are never eligible for tuning promotion or final reports.

Preview the exact eight-cell-per-seed matrix without starting compute:

```bash
python scripts/run_primary_matrix.py \
  --config configs/experiments/vcod_primary_2x2.yaml \
  --datasets moca_mask_dense camotion \
  --systems DS VI DT VV --seeds 42 43 44 --dry-run
```

Train/evaluate an individual probe after dataset and backbone gates pass:

```bash
python scripts/train_probe.py --config configs/experiments/vcod_primary_2x2.yaml \
  experiment.system_id=DT dataset.name=moca_mask_dense --smoke
python scripts/evaluate.py --run-dir outputs/<run_id> --split test --save-logits
```

The evaluator writes float logits/raw sigmoid/min-max views separately,
`per_frame.csv`, `per_video.csv`, and `summary.json`. Build the paired report
only after all requested primary cells exist:

```bash
python scripts/summarize_results.py --runs-root outputs \
  --matrix configs/experiments/vcod_primary_2x2.yaml \
  --temporal-sampling-ablation configs/experiments/moca_temporal_sampling.yaml \
  --output outputs/reports/primary_comparison \
  --attributes OC OV MB SO MO UE SC BO
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
