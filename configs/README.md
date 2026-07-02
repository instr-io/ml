# Preset Configs

This folder intentionally tracks only the two preset architectures we actually
use:

- `v1.json`: bootstrap-style worker-compatible Mamba preset
- `mamba3.json`: public Mamba-3 training preset

These preset JSON files do not commit machine-specific runtime paths.

Provide dataset/output/checkpoint locations at runtime via either:

- `.env` / exported environment variables:
  - `INSTR_DATA_DIRS`
  - `INSTR_OUTPUT_DIR`
  - `INSTR_CHECKPOINT_PATH`
- CLI overrides:
  - `--data_dirs`
  - `--output_dir`
  - `--checkpoint`

Training still writes a run-specific `config.json` next to checkpoints under the
output directory. That runtime artifact is expected and is separate from these
checked-in preset configs.
