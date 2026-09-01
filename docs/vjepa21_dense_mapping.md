# V-JEPA2.1 dense mapping lock

The official V-JEPA2.1 hub constructor fixes 384×384 inputs, 16×16 spatial patches, 64 video frames, tubelets of 2 frames, RoPE, and `img_temporal_dim_size=1`. The study therefore locks `T=64`, stride 1, and source target index 32. At 30 FPS this spans approximately 2.1 seconds; each run records the actual source FPS and duration.

The image baseline calls the encoder with `[B,C,1,H,W]`, activating the official image pathway. It is never emulated with repeated video frames.

For video input `[B,C,64,384,384]`, patch embedding yields a grid of `32×24×24 = 18,432` dense tokens. The verified reshape contract is temporal-major flattening:

`token_index = ((tau * 24) + row) * 24 + column`.

Temporal token `tau` covers source clip indices `[2*tau, 2*tau+1]`. Target source index 32 maps to temporal token 16, covering source indices 32–33. This asymmetric even-tubelet rule is explicit and common to VV and VR. Runtime code rejects unexpected token counts rather than guessing.

This mapping is based on the official hub/model implementation and must still receive the manual semantic sign-off required by the plan on the exact local checkpoint before an official test run.
