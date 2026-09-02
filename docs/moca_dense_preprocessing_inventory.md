# MoCA dense preprocessing inventory

This document records the pre-patch implementation and the integration points for the immutable `moca_mask_dense_v1` preprocessing product.

## Existing behavior

- `src/cod_ssl/data/moca_mask.py` is a thin `ManifestVideoCODDataset` wrapper. It does not verify release provenance or require a derived-build manifest.
- `src/cod_ssl/data/vcod_bootstrap.py` interprets `MoCA-Mask-Pseudo` as the full video release. It exposes 19,313 train and 3,626 test frames, uses pseudo masks for every training frame, and uses manual masks only for validation/test targets. This is not the corrected scientific protocol.
- Public `Imgs` targets are not currently treated as a complete video by the adapter itself, but the bootstrap substitutes the pseudo-label release as if it supplied authoritative dense context.
- `frame_number` and manifest-row sequence position are conflated. `ClipSampler` strides through ordered rows and does not independently validate source-frame spacing.
- The canonical runtime manifest is CSV. It has target/context rows through `is_target`, but lacks benchmark/source identity separation, legal subsequence bounds, release provenance, and manifest hashes.

## Reusable infrastructure

- `cod_ssl.data.bootstrap.IMAGE_SUFFIXES` provides supported image suffixes.
- `cod_ssl.data.vcod_bootstrap.sha256_with_progress` and `cod_ssl.utils.run.file_sha256` provide existing hashing helpers.
- `cod_ssl.data.video_manifest.assert_disjoint_video_splits` checks source-video leakage, but the dense build also needs benchmark-subsequence and source-frame-key checks.
- `cod_ssl.data.clip_sampler.ClipSampler` provides deterministic replicate padding. It needs a source-frame-aware path for D1/S5 sampling inside legal benchmark ranges.
- Archive extraction utilities exist, but there is no reusable symlink/hardlink/copy publication helper. The new preprocessor will implement narrowly scoped, verified materialization and atomic publication.

## Required consumers

- New importable preprocessing modules will inventory Original MoCA and public MoCA-Mask, apply only exact/explicit sequence mappings, verify all target RGB hashes, resolve `manual_target_hull_v1`, serialize deterministic manifests, materialize an optional lightweight view, and verify published builds.
- `scripts/prepare_moca_mask_dense.py` will be a thin CLI supporting dry-run, verify-only, materialization selection, boundary policy, and explicit overwrite.
- `ManifestVideoCODDataset` and the MoCA adapter will consume the canonical dense frame/target semantics and reject raw sparse MoCA-Mask for dense runs.
- `scripts/inspect_dataset.py`, primary/ablation matrix scripts, training/evaluation output, summarization, and notebooks 05–07 must use `moca_mask_dense` and record cadence/source span.
- The scientific source of truth is the generated manifest directory, not filename or directory heuristics in training code.

## Locked release expectations

- Original MoCA: 141 zero-based consecutive sequences and 37,250 JPEG frames, without pixel masks.
- Public MoCA-Mask: 87 benchmark sequences, official 71/16 split, 4,691 target RGB files, and 4,691 manual masks; it contains no dense intermediate RGB release.
- Five aliases are explicit: `snow_leopard_4.1/.2 -> snow_leopard_4` and `snow_leopard_5.1/.2/.3 -> snow_leopard_5`.
- The default legal range is each benchmark sequence's inclusive manual-target hull. Its computed dense-frame total is reported and is never forced to the paper-described 22,939 frames.
