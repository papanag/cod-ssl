# Patch 2 release-semantics inventory

This inventory records the repository state before the Patch 2 functional changes. It distinguishes paper-reported metadata, inspected public-release facts, active assumptions, fixtures, wording, and pre-existing artifacts. Existing run artifacts are not relabeled.

## CAMotion

| Location | Current meaning | Classification | Required action |
|---|---|---|---|
| `configs/datasets/camotion.yaml` | `frames_total: 149319`, `annotated_frames_total: 30028` | active adapter/config assumption | Replace the ambiguous total with an immutable `camotion_public_stride5_v1` release profile: 30,028 unique public RGB/GT pairs, source step 5, no dense intermediates; retain 149,319 only as paper-described collected frames. |
| `src/cod_ssl/data/camotion_bootstrap.py` | Sequence-organized RGB/GT are canonical; 359/115 and 23,253/6,775 are validated | discovered public-release metadata and active adapter behavior | Preserve canonical selection, add explicit sequence/source positions and cadence fields, validate source step five, and verify flattened exports when the full archive is available. |
| `scripts/bootstrap_vcod_data.py` | Selectively extracts only sequence-organized CAMotion trees | active bootstrap behavior | Preserve canonical extraction; record the immutable release profile and full-archive duplication audit status. |
| `scripts/inspect_dataset.py`, `docs/camotion_protocol.md`, `README.md` | Public archive exposes 30,028 sequence RGB/GT pairs while 149,319 describes unavailable collected frames | report wording | Retain the distinction and replace “dense discrepancy” language with explicit stride-5 release semantics. |
| CAMotion tests | Fixtures exercise sparse released observations and manual targets | test fixture expectations | Make source/released-step semantics explicit; add rejection of invented intermediates and flattened samples. |

## MoCA

| Location | Current meaning | Classification | Required action |
|---|---|---|---|
| `configs/datasets/moca_mask.yaml` | Sparse public MoCA-Mask is the active dataset, with no release profile | active config assumption | Retire from the primary matrix; retain only as explicitly named legacy/public-sparse validation if needed. Add `moca_mask_dense.yaml`. |
| `src/cod_ssl/data/vcod_bootstrap.py` | Combines MoCA-Mask manual targets with `MoCA-Mask-Pseudo`; training treats pseudo masks as targets and the pseudo release as full context | active adapter assumption | Remove from the primary path. Dense context must come only from byte-verified Original MoCA frames; all 4,691 scientific targets remain manual. |
| `scripts/bootstrap_vcod_data.py` | Downloads the public manual archive plus the pseudo-label archive and emits `moca_mask.csv` | active bootstrap assumption | Replace the pseudo-release build with an explicit Original-MoCA + MoCA-Mask preprocessing command. Do not preprocess automatically in training jobs. |
| `src/cod_ssl/data/video_manifest.py` | `frame_number` is both chronological source identity and effective sequence position; clips stride over manifest rows | clip-cadence semantics | Add separate source frame number and legal sequence position. Dense MoCA source-stride sampling must operate in source-frame units and remain inside benchmark subsequence bounds. |
| Primary matrix/notebooks/docs | Dataset ID is `moca_mask`; cadence is called `default` | active protocol and report wording | Change the primary dataset ID to `moca_mask_dense`, expose D1 metadata, and add the separately trained S5 temporal-sampling/coverage runs for DT/VV. |
| MoCA tests | Expect 19,313/3,626 pseudo-release frames and pseudo training targets | test fixture expectation | Replace with release inventory, explicit alignment, boundary, deterministic-manifest, and synthetic end-to-end tests. |

## Shared schema and consumers

The current CSV manifest requires dataset, split, video/source IDs, frame ID/number, image/mask paths, and annotation type. It supports `is_target`, but does not require release profile, benchmark/source identity separation, sequence position, cadence, boundary policy, dense-context availability, or preprocessing hashes. `ManifestVideoCODDataset`, training/evaluation scripts, result summaries, matrix expansion, and Colab notebooks consume this contract.

Patch 2 will add fields compatibly where possible. `frame_number` remains a backward-compatible alias for source frame number; new code will not infer released position from it. Old artifacts remain immutable and must either be read under their prior schema or rejected with a clear semantic-profile error.
