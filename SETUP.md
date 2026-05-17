# Setup

This file is the operational setup guide for `ml/`.

## Ubuntu Packages

Install the likely system packages first:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 build-essential python3-dev pkg-config git git-lfs ninja-build
```

Notes:

- `ffmpeg` is required for audio conversion paths.
- `libsndfile1` is required by `soundfile`.
- `ninja-build` helps with `mamba-ssm` and `causal-conv1d` builds.

## Python

Known-good runtime on the deployed worker is Python `3.11`.

Install the default dependencies with:

```bash
pip install -r requirements.txt
```

For an exact known snapshot, use:

```bash
pip install -r requirements-pinned.txt
```

Optional:

```bash
pip install wandb
pip install boto3
```

## CUDA Extensions

The main setup pain point is usually `mamba-ssm` and `causal-conv1d`.

If a plain `pip install -r requirements.txt` works for your machine, keep it.
If those two packages are slow to build or fail to build, install them
explicitly with your GPU architecture limited via `TORCH_CUDA_ARCH_LIST`.

First, inspect the GPU:

```bash
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Examples:

- NVIDIA A40 -> `TORCH_CUDA_ARCH_LIST="8.6"`
- NVIDIA GeForce RTX 5060 Ti -> `TORCH_CUDA_ARCH_LIST="12.0"`

If CUDA is installed in a non-default location, export `CUDA_HOME` first.
Example:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
```

Recommended source-build command:

```bash
pip install packaging ninja
TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=32 \
  pip install --no-cache-dir --no-build-isolation \
  --no-binary causal-conv1d --no-binary mamba-ssm \
  causal-conv1d mamba-ssm
```

Notes:

- `--no-build-isolation` matters. Without it, pip may build against a different
  PyTorch environment.
- `MAX_JOBS=32` is optional, but it can speed up builds on bigger machines.
- Change `TORCH_CUDA_ARCH_LIST` to match the actual GPU on the machine.

## Sanity Checks

After install, verify the core stack:

```bash
python - <<'PY'
import torch
import torchaudio
import numpy
import scipy
import soundfile
import tqdm
import mamba_ssm
import causal_conv1d

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchaudio", torchaudio.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("soundfile", soundfile.__version__)
print("tqdm", tqdm.__version__)
print("mamba_ssm", mamba_ssm.__version__)
print("causal_conv1d", causal_conv1d.__version__)
PY
```

Optional package checks:

```bash
python - <<'PY'
for name in ["wandb", "boto3"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "installed"))
    except ImportError:
        print(name, "not installed")
PY
```

## Environment

Copy `.env.example` to `.env` and fill in what you need:

```bash
cp .env.example .env
```

Important variables:

- `INSTR_DATA_DIRS`
- `INSTR_OUTPUT_DIR`
- `INSTR_CHECKPOINT_PATH`
- `INSTR_ENABLE_WANDB`
- `INSTR_WANDB_PROJECT`
- `INSTR_WANDB_ENTITY`
- `INSTR_WANDB_API_KEY`
- `INSTR_ENABLE_S3`
- `INSTR_AWS_REGION`
- `INSTR_S3_BUCKET`
- `INSTR_S3_CHECKPOINT_PREFIX`

## First Commands

Train:

```bash
python -m training.train
```

For long training runs, prefer a named `tmux` or `screen` session so you can
reconnect later and inspect progress.

Recommended `tmux` flow:

```bash
tmux new -s ml-train
python -m training.train 2>&1 | tee train.log
```

Useful `tmux` commands:

```bash
tmux attach -t ml-train
tmux capture-pane -pt ml-train | tail -n 50
tmux ls
```

`screen` alternative:

```bash
screen -S ml-train
python -m training.train 2>&1 | tee train.log
```

Useful `screen` commands:

```bash
screen -r ml-train
screen -ls
```

## Checkpoints And Resume

Training writes checkpoints under:

```text
./outputs/<experiment_name>/
```

Typical files:

- `config.json`
- `latest.pt`
- `best_model.pt`
- `step_000050.pt`, `step_000100.pt`, ...

The safest resume flow is to reuse the saved `config.json` from the run and
point `--checkpoint` at `latest.pt`:

```bash
python -m training.train \
  --config ./outputs/vocal_separator_base/config.json \
  --checkpoint ./outputs/vocal_separator_base/latest.pt
```

If you only want to use the checkpoint path via environment, you can also set:

```bash
INSTR_CHECKPOINT_PATH=./outputs/vocal_separator_base/latest.pt
```

but for training resume, `--checkpoint` is clearer.

To resume inside a named `tmux` session:

```bash
tmux new -s ml-train-resume
python -m training.train \
  --config ./outputs/vocal_separator_base/config.json \
  --checkpoint ./outputs/vocal_separator_base/latest.pt \
  2>&1 | tee resume.log
```

## Adding More Data Later

The trainer accepts multiple dataset roots as a comma-delimited list.
Each root should contain paired folders like:

```text
root_a/song_001/in.wav
root_a/song_001/out.wav
root_b/song_002/in.wav
root_b/song_002/out.wav
```

If a root already has explicit split folders, use:

```text
root_a/train/song_001/in.wav
root_a/train/song_001/out.wav
root_a/val/song_002/in.wav
root_a/val/song_002/out.wav
root_a/test/song_003/in.wav
root_a/test/song_003/out.wav
```

When `train/` and `val/` are present, the trainer uses them directly. Otherwise
it falls back to a stable per-pair hash split for validation.

You can start with one folder:

```bash
INSTR_DATA_DIRS=./data python -m training.train
```

and later resume from the same checkpoint while adding more folders:

```bash
INSTR_DATA_DIRS=./data,./data_curriculum,./data_edge_cases \
  python -m training.train \
  --config ./outputs/vocal_separator_base/config.json \
  --checkpoint ./outputs/vocal_separator_base/latest.pt
```

You can do the same with `--data_dirs`:

```bash
python -m training.train \
  --config ./outputs/vocal_separator_base/config.json \
  --checkpoint ./outputs/vocal_separator_base/latest.pt \
  --data_dirs ./data,./data_curriculum,./data_edge_cases
```

That is the intended way to layer in harder examples, curriculum folders, or
new edge-case data over time without starting over from scratch.

Run local inference:

```bash
python -m inference.infer input.wav output.wav --checkpoint ./outputs/your_run/latest.pt
```
