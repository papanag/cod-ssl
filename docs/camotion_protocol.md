# CAMotion protocol and provenance

CAMotion is the second active VCOD dataset and is accepted by the primary matrix
and completeness gate.

## Official sources

- Dataset/repository: `https://github.com/Garyson1204/CAMotion`
- Dataset Google Drive file: `1YzNdlDhsfgXTZ-Ya1w9wn3SjTXwU2xFs`
- Metadata commit: `bf92692f9f9f2820185f9aa9a06fd2891dadf9a7`
- `attributes_per_sequence.txt` SHA-256:
  `6ad95102a836ef5a199e6e0a642ee7ddfbf2f6d8065c40742014cfab934abcd9`
- Usage: academic research only; commercial use requires author permission.

The bootstrap selects only `TrainDataset_per_sq` and `TestDataset_per_sq` from the
archive after verifying their CRC32/size multisets. Flattened `CAMotion-TR` and
`CAMotion-TE` exports duplicate the annotated assets and are never sampled. The selected release has 359/115 sequences and
23,253/6,775 paired RGB/manual-GT targets. Although the paper/project reports
149,319 collected source frames, the public archive contains only every fifth source frame.
No frames or masks are interpolated or manufactured. Available annotated RGB
frames remain chronologically ordered source-stride-5 context for the locked common sampler.

## Attribute interpretation

`MO`, `BO`, `SO`, `UE`, `OC`, `SC`, `OV`, and `MB` are overlapping sequence-level
labels. Every emitted target carries all eight explicit Boolean values and
`attribute_scope=sequence`. Empty official rows produce eight `false` values;
missing rows are fatal. `foreground_fraction` is a separate continuous per-frame
diagnostic and never replaces official `SO` or `BO` labels.

## Official evaluator parity

`scripts/check_camotion_parity.py` mirrors the official `eval_video.py` global
accumulation using the repository's metric wrapper and fixed saved 8-bit maps. It
checks S-measure, weighted F-measure, MAE, adaptive E-measure, and mean E-measure.
The study's evaluator remains authoritative and applies per-image min-max
normalization before uint8 quantization. `E_max` is documented separately because
the study averages each frame's maximum threshold score, while the official script
takes the maximum after averaging threshold curves. Video-weighted study-primary
results and frame-weighted official-compatible results are always labeled
separately.
