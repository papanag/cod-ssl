# Dataset validation sign-off

The automated inspection pipeline is implemented in `scripts/inspect_dataset.py`.
Notebook 05 bootstraps Original MoCA and public MoCA-Mask into Drive, creates the
verified `moca_mask_dense_v1` product, and then runs inspection. The following
human visual-review gates still remain open.

| Dataset/regime | Release | Reviewer | Date | 20 overlays | boundary clips | source pairing | Status |
|---|---|---|---|---|---|---|---|
| MoCA-Mask + Original MoCA dense context | official releases; `manual_target_hull_v1` | pending | pending | pending | pending | SHA-256 target alignment | ready to bootstrap/review |
| CAMotion | official 2026 release; academic research only | pending | pending | pending | pending | sequence-level attributes | ready to bootstrap/review |

CAMotion release note: the official public ZIP contains 359 training and 115 test
sequence directories with 23,253 and 6,775 paired RGB/manual-GT targets. It also
contains flattened duplicate image exports, which the adapter excludes. The
paper-described 149,319 frames are collected source material rather than public
dense context; the canonical manifest contains 30,028 released source-stride-5
RGB frames. A reviewer must acknowledge this before training.

MoCA sign-off must record both release hashes, build ID, preprocessing-manifest
hash, derived dense-frame count, boundary policy, reviewer/date, discrepancies,
and the limitation that the derived view is not an exact reconstruction of the
unavailable paper-described 22,939-frame package.
The official mask PNGs are retained byte-for-byte and their stored grayscale
values are recorded in `mask_quality.csv`; training and evaluation use the
repository-wide nonzero-foreground rule rather than requiring 0/255 storage.
The initial dense build hashes every linked target RGB and mask. Routine cached
bootstraps validate manifest checksums and structure without rereading all 4,691
target pairs from Drive. Run `scripts/prepare_moca_mask_dense.py --verify-only`
with the configured `--output-root` whenever a fresh full asset-integrity check
is required.

CAMotion sign-off must record the archive SHA-256, pinned attribute-file SHA-256,
official repository commit, exact validation IDs, discrepancies, exclusions,
reviewer, and date. Challenge attributes overlap and have sequence scope.

No official scientific test run may begin until the relevant row is signed off and all warnings in the generated inspection report are explicitly acknowledged.
