# Pi05 HBM artifact and board contract

This document records the compiled artifacts without checking binary model
files into Git. The artifacts are produced on the RTX 4090 in the
`pi05_qat`/`leap_llm` environment and live on shared storage.

The current shared-storage fingerprints are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `pi05_siglip_ptq.hbm` | 444684936 | `f5c418121e83711919d515a07379aa63dc9a6cb9ca65125aa02bf6be3825dadf` |
| `pi05_gemma_llm_ptq.hbm` | 3090784728 | `69c5f03148aa478086cbba6d7f4517a2f282cdf4c4b856a37109396cf8967a05` |
| `pi05_gemma_expert_ptq.hbm` | 462292008 | `5a89ea25f50ee3c04720fe5e7b2a468826485b0018dfec922aa45d4392682d06` |

## Artifact set

The complete Pi05 graph set is:

```text
pi05_siglip_ptq.hbm          # vision/SigLIP, 136 visual tokens
pi05_gemma_llm_ptq.hbm      # language prefix + KV-cache outputs
pi05_gemma_expert_ptq.hbm   # action expert / denoising step
```

The 4090-side HBM header inspection shows the following graph contracts:

- SigLIP takes image `[1,3,224,224]` (FP16) and token IDs `[1,256]` (INT64),
  and returns visual tokens `[1,136,2048]` (FP16).
- LLM takes token IDs `[1,200]` (INT32), visual prefix `[1,408,2048]`
  (FP16), image features `[1,1,608,608]` (FP16), position IDs `[1,608]`
  (INT32), and a float mask `[1,608]`; it returns `[1,608,2048]` plus 36
  KV-cache tensors of shape `[1,608,256]`.
- Expert takes state `[1,32]`, noisy actions `[1,50,32]`, timestep `[1]`,
  prefix mask `[1,1,50,658]`, position IDs `[1,50]`, and 36 KV-cache tensors
  of shape `[1,608,256]`; it returns velocity `[1,50,32]`.
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
