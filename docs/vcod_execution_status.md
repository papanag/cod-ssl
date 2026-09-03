# Focused VCOD execution status

Status date: 2026-09-02.

## Completed implementation gates

- Static COD metric behavior is frozen by a deterministic regression fixture, and the supplied static result table is stored in a machine-readable registry.
- Canonical video data contract, video-disjoint leakage checks, clip sampler, boundary validity, and shared clip geometry are implemented.
- Derived MoCA-Mask and CAMotion use one canonical-manifest adapter with explicit source-frame cadence.
- DS/VI use one target frame. VI invokes the official V-JEPA image path. VV/VR invoke the native video path. DT/DM use framewise frozen DINO features.
- Dense V-JEPA tubelet mapping is strict and documented; unexpected token counts fail.
- Official GMMix source provenance is pinned. The one-layer capacity adaptation is declared before test evaluation.
- The shared projection/two-block decoder, BCE+Dice loss, gradient audit, float prediction writer, paired video bootstrap, timer, matrix launcher, and report generator are implemented.
- Automated tests pass without requiring gated checkpoints or private datasets.
- Original MoCA and public MoCA-Mask now have a resumable, progress-reporting Drive
  bootstrap and a separate immutable preprocessing product. It verifies all 4,691
  targets by SHA-256, exposes only legal consecutive source frames, and uses manual
  masks exclusively.
- CAMotion's official archive, official 359/115 split, 30,028 manual targets,
  and pinned 474-row sequence-attribute metadata have a strict bootstrap and
  validation path.

## Open manual/external gates

These are not implementation defects and cannot be completed from the repository alone:

1. Run notebook 05 to bootstrap Original MoCA plus MoCA-Mask, verify the derived build,
   then inspect it, acknowledge warnings, and sign
   the overlay/boundary review. The validation-video IDs are serialized automatically.
2. Bootstrap and manually inspect CAMotion, acknowledge that its public archive
   contains 30,028 unique observations at source-frame step five, and sign the sequence-attribute gate.
3. Supply the approved DINOv3 and V-JEPA2.1 ViT-B/16 checkpoints and official repository clones; record their SHA-256 hashes and commits.
4. Run the V-JEPA image/video inspection on the exact checkpoint and manually sign target-tubelet and spatial-orientation semantics.
5. Install `mamba-ssm` in the CUDA environment and run DT mixed-precision smoke training.
6. Complete the declared 250/1,000/3,000-step successive-halving learning-rate
   selection on training/validation videos, review its immutable promotion receipts,
   freeze the resulting protocol, then run clean official test seeds.
7. Complete all eight primary cells and the two additional MoCA S5 temporal cells per
   seed. Only afterward run DM/VR, shuffled diagnostics, efficiency modes, and qualitative review.

No dataset inspection, checkpoint verification, training result, or scientific metric has been fabricated. The report generator intentionally fails when a requested primary cell or paired source-video key is missing.
