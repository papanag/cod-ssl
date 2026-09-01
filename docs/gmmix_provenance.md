# GatedMambaMix provenance

- Paper: *Towards Data-Efficient Video Pre-training with Frozen Image Foundation Models*, Orlova, Cavagnero, and Dubbelman, CVPR Workshops 2026, [arXiv:2605.19137](https://arxiv.org/abs/2605.19137).
- Reference repository: [tue-mps/towards-video-image-frozen](https://github.com/tue-mps/towards-video-image-frozen).
- Pinned reference commit: `59249bf83311bc34bae277e2e8adec287ffe5d0f`.
- Source consulted: `models/ssm_modules/mamba_modules.py`, specifically `GatedMambaMixSeqCore` and its spatial/Mamba blocks.
- License: MIT, copyright 2026 Mobile Perception Systems Lab at TU/e. The notice is retained in the adapted module.
- Local source: `src/cod_ssl/temporal/gated_mamba_mix.py`.

## Adaptation and locked capacity

The local adapter keeps the reference ordering: per-frame spatial self-attention and MLP, per-coordinate temporal Mamba residual, and a learned sigmoid interpolation between pre- and post-Mamba representations. It converts the repository contract `B,T,C,H,W` to the reference `B,T,N,D` layout and returns the explicitly selected target frame as a dense grid.

The initial small VCOD configuration uses one GMMix layer, `d_state=16`, expansion 2, 12 spatial heads, MLP ratio 4, dropout 0.1, and a direct `2D -> D` gate. This is intentionally fixed before test evaluation. It differs from the reference paper inference helper's four-layer task configuration and must be reported as a capacity adaptation, not an exact reproduction of that experiment.

`mamba-ssm` is an optional CUDA-environment dependency. Boundary validity is checked but not inserted into the state-space recurrence, because doing so would be an unvalidated architecture change. Primary accuracy uses independent window mode; no state crosses samples or videos.
