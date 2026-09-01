# Dataset validation sign-off

The automated inspection pipeline is implemented in `scripts/inspect_dataset.py`. Official dataset releases and canonical manifests are not stored in this repository, so the following required manual gates remain open until those assets are supplied locally.

| Dataset/regime | Release | Reviewer | Date | 20 overlays | boundary clips | source pairing | Status |
|---|---|---|---|---|---|---|---|
| MoCA-Mask | pending | pending | pending | pending | pending | n/a | blocked on local release |
| CamoVid60K small | pending | pending | pending | pending | pending | pending | blocked on local release |
| CamoVid60K large | pending | pending | pending | pending | pending | pending | blocked on local release |

No official scientific test run may begin until the relevant row is signed off and all warnings in the generated inspection report are explicitly acknowledged.
