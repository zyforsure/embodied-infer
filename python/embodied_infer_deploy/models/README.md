# Model implementations

Each model family has an isolated directory and implements the contracts in
`core/contracts.py`.

| Registry name | Directory | Target |
| --- | --- | --- |
| `mock` | `mock/` | local tests and protocol smoke tests |
| `turbovla-tensorrt` | `turbovla/` | RTX 4090 / Jetson AGX Orin |
| `turbovla-s600-hbm` | `turbovla/` | direct S600 HBM runtime |
| `turbovla-s600-remote` | `turbovla/` | existing S600 server gateway |

Use `_template/` for a new model family. Heavy model dependencies must be
imported inside the selected backend so registry discovery remains portable.
