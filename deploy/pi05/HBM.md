# Pi05 HBM artifact and board contract

This document records the compiled artifacts without checking binary model
files into Git. The artifacts are produced on the RTX 4090 in the
`pi05_qat`/`leap_llm` environment and live on shared storage.

## Artifact set

The complete Pi05 graph set is:

```text
pi05_siglip_ptq.hbm          # vision/SigLIP, 136 visual tokens
pi05_gemma_llm_ptq.hbm      # language prefix + KV-cache outputs
pi05_gemma_expert_ptq.hbm   # action expert / denoising step
```

The recorded compile logs show the following graph contracts:

- LLM input tensors include token IDs `[1,200]`, visual prefix `[1,408,2048]`,
  image features `[1,1,608,608]`, position IDs `[1,608]`, and a float mask;
  it returns the prefix hidden state plus 36 KV-cache tensors.
- Expert input tensors include state `[1,32]`, noisy actions `[1,50,32]`, a
  timestep scalar, prefix mask `[1,1,50,658]`, and 36 KV-cache tensors; it
  returns `[1,50,32]` velocity.
- The native policy head is 16 action dimensions. The shared runtime adapter
  projects the native result to the configured 14D model contract and then
  reconstructs the raw 18D robot command.

The exact tensor names and dtypes must be queried from the target board's HBRT
loader; do not infer them from file names. A file-size check is not an
execution check.

## Source-side verification

On the 4090, use the environment that contains `leap_llm` and verify the
artifacts and compile logs:

```bash
test -f /shared-data/zzyy/quantation/hbm_model/pi05_siglip_ptq.hbm
test -f /shared-data/zzyy/quantation/hbm_model/pi05_gemma_llm_ptq.hbm
test -f /shared-data/zzyy/quantation/hbm_model/pi05_gemma_expert_ptq.hbm
tail -n 20 /shared-data/zzyy/quantation/pi05_hbm/llm_int8/compile.log
tail -n 20 /shared-data/zzyy/quantation/pi05_hbm/expert_int8/compile.log
```

Do not put SSH passwords in these commands or in the repository.

## Board acceptance gate

Pi05 is considered deployed on a board only after all of the following are
recorded for that board:

1. The three HBM files are copied with SHA-256 verification.
2. HBRT loads each graph and reports input/output names, shapes, and dtypes.
3. One synthetic finite-input request executes the vision -> LLM -> expert
   chain and returns finite `[1,16,16]` native actions (or an explicitly
   documented equivalent layout).
4. The Pi05 service accepts the canonical embodied-infer request and a
   RoboTwin smoke episode completes with finite reconstructed 18D actions.

The current Orin and `192.168.10.252` services do not satisfy this gate: they
currently expose TurboVLA contracts. Keep those services unchanged until a
Pi05-specific HBRT server is installed and validated.
