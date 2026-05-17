# instr.io

Open-source vocal separation model used by `instr.io`.

This repo contains a band-split BiMamba U-Net that takes a stereo mix and
predicts a stereo instrumental. For installation, environment variables,
dataset layout, training commands, resume, and inference, start with
[SETUP.md](SETUP.md).

## Overview

The model is a band-split BiMamba U-Net.

- frequency bins are grouped into non-uniform bands inspired by BSRNN
- the encoder is a 3-level U-Net with bidirectional Mamba blocks
- the bottleneck alternates Mamba and self-attention
- the decoder uses cross-attention into a pooled memory bank plus gated skips
- the model predicts a complex mask over the input STFT and inverts it back to audio

The current default config in [model/config.py](model/config.py) uses:

- `sample_rate=44100`
- `n_fft=2048`
- `hop_length=256`
- `chunk_seconds=3.0`
- `d_model=256`
- `n_heads=4`
- `n_encoder_layers=4`
- `n_decoder_layers=4`
- `n_bottleneck_layers=6`
- `d_state=32`
- `use_mid_side=false`

## STFT And Band Split

A `2048`-point STFT produces `1025` frequency bins. The model then groups them
into `72` non-uniform bands in [model/arch/band_split.py](model/arch/band_split.py),
with more resolution in the vocal range, especially roughly `80 Hz` to `4 kHz`.

Each band is embedded from:

- left real
- left imaginary
- right real
- right imaginary

into a learned `d_model`-dimensional representation, plus a learned band
position embedding.

## U-Net

The architecture is a 3-level U-Net in [model/arch/separator.py](model/arch/separator.py).
Time is downsampled by `2x` at each encoder level, so the bottleneck runs at
`T/8`. Skip connections preserve higher-resolution detail for the decoder.

## Mamba Encoder

The encoder uses dual-path bidirectional Mamba blocks from
[model/arch/mamba_blocks.py](model/arch/mamba_blocks.py).

Each encoder block mixes:

- time within each band
- bands within each time step

The earlier encoder levels also include local temporal convolution before the
dual-path Mamba block.

## Bottleneck And Memory Bank

At the bottleneck, the model alternates dual-path BiMamba blocks with global
self-attention from [model/arch/attention_blocks.py](model/arch/attention_blocks.py).
This is the first stage where bands and time are flattened together for global
mixing.

The decoder does not attend back to every encoder token directly. Instead,
`MultiScaleMemoryBank` pools each encoder level into a fixed number of summary
tokens per band:

- level 1: `64`
- level 2: `48`
- level 3: `32`
- bottleneck: `16`

This gives `160` pooled memory tokens per band for decoder retrieval.

## Decoder

The decoder mirrors the encoder and upsamples time back to the original
resolution.

Each `DecoderBlock` combines:

- factorized band/time cross-attention into encoder memory
- gated skip fusion from the matching encoder level
- SwiGLU feed-forward mixing

The cross-attention is structured rather than fully flattened:

- time attention runs within each band against pooled encoder time tokens
- band attention runs across bands using local time-aligned encoder summaries

Skip connections pass through `GatedSkipFusion`, so the decoder learns how much
encoder detail to reintroduce instead of copying skips directly.

## Mask Head

`BandMerge` in [model/arch/band_split.py](model/arch/band_split.py) projects
decoder features back to the original frequency layout and predicts four values
per frequency bin:

- `L_mag`
- `L_phase`
- `R_mag`
- `R_phase`

Magnitude is bounded with `sigmoid`. Phase adjustment is bounded with
`tanh * pi`. The resulting complex masks are multiplied against the input STFT,
then inverted back to waveform with ISTFT.

## Training Objective

The current default loss stack in [training/losses.py](training/losses.py)
uses three terms:

- magnitude loss
- band-wise log-magnitude loss
- SI-SDR loss

## Code Map

- [model/arch/separator.py](model/arch/separator.py): top-level model
- [model/arch/band_split.py](model/arch/band_split.py): band layout and mask head
- [model/arch/mamba_blocks.py](model/arch/mamba_blocks.py): encoder blocks
- [model/arch/attention_blocks.py](model/arch/attention_blocks.py): bottleneck attention, memory bank, decoder attention, gated skips
- [training/losses.py](training/losses.py): default loss stack
- [SETUP.md](SETUP.md): installation, env vars, data layout, training, resume, and inference
