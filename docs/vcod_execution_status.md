# Focused VCOD execution status

Status date: 2026-09-01.

## Completed implementation gates

- Static COD metric behavior is frozen by a deterministic regression fixture, and the supplied static result table is stored in a machine-readable registry.
- Canonical video data contract, video-disjoint leakage checks, clip sampler, boundary validity, and shared clip geometry are implemented.
- MoCA-Mask and CamoVid60K use one canonical-manifest adapter so clip behavior cannot diverge silently.
- DS/VI use one target frame. VI invokes the official V-JEPA image path. VV/VR invoke the native video path. DT/DM use framewise frozen DINO features.
- Dense V-JEPA tubelet mapping is strict and documented; unexpected token counts fail.
- Official GMMix source provenance is pinned. The one-layer capacity adaptation is declared before test evaluation.
- The shared projection/two-block decoder, BCE+Dice loss, gradient audit, float prediction writer, paired video bootstrap, regime interaction, timer, matrix launcher, and report generator are implemented.
- Automated tests pass without requiring gated checkpoints or private datasets.

## Open manual/external gates

These are not implementation defects and cannot be completed from the repository alone:

1. Obtain and identify the exact official MoCA-Mask and CamoVid60K releases; create canonical manifests without changing official test membership.
2. Run inspection for each regime, acknowledge warnings, sign the 20-overlay and boundary-clip review in `docs/dataset_validation.md`, and serialize the approved validation-video IDs.
3. Supply the approved DINOv3 and V-JEPA2.1 ViT-B/16 checkpoints and official repository clones; record their SHA-256 hashes and commits.
4. Run the V-JEPA image/video inspection on the exact checkpoint and manually sign target-tubelet and spatial-orientation semantics.
5. Install `mamba-ssm` in the CUDA environment and run DT mixed-precision smoke training.
6. Perform equal-scope learning-rate selection on training/validation videos, freeze the resulting protocol, then run official tests.
7. Complete all 12 primary cells for each declared seed. Only afterward run DM/VR, bootstrap reports, efficiency modes, and qualitative review.

No dataset inspection, checkpoint verification, training result, or scientific metric has been fabricated. The report generator intentionally fails when a requested primary cell or paired source-video key is missing.
