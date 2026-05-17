"""
Training loop for vocal separator.

Supports:
- Mixed precision training
- Gradient accumulation
- Checkpointing
- W&B Logging
- tqdm progress bars
- Audio sample generation
"""

import logging
import random
from pathlib import Path
from typing import Optional, Dict

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.arch.separator import create_model, count_parameters
from model.dataset import collate_fn
from model.config import Config, RuntimeConfig, base_config, load_runtime_config
from training.data import build_train_val_datasets
from training.losses import SeparationLoss
from training.observability import build_progress_postfix, build_train_metrics

try:
    import boto3
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training manager for vocal separator."""

    def __init__(self, config: Config, runtime: Optional[RuntimeConfig] = None):
        self.config = config
        self.runtime = runtime or load_runtime_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = self.device.type == "cuda"
        set_seed(config.seed)

        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.samples_dir = self.output_dir / "samples"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        config.save(self.output_dir / "config.json")

        # Build model
        self.model = create_model(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            sr=config.audio.sample_rate,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_encoder_layers=config.model.n_encoder_layers,
            n_decoder_layers=config.model.n_decoder_layers,
            n_bottleneck_layers=config.model.n_bottleneck_layers,
            dropout=config.model.dropout,
            use_mid_side=getattr(config.model, 'use_mid_side', False),
            d_state=getattr(config.model, 'd_state', 32),
        ).to(self.device)

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model parameters: {count_parameters(self.model) / 1e6:.2f}M")
        self.model_unwrapped = self.model

        # Build loss
        self.loss_fn = SeparationLoss(
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            mag_weight=config.loss.mag_weight,
            mr_stft_weight=config.loss.mr_stft_weight,
            si_sdr_weight=config.loss.si_sdr_weight,
            spectral_sdr_weight=config.loss.spectral_sdr_weight,
            band_weight=config.loss.band_weight,
        ).to(self.device)

        self.train_dataset, self.val_dataset, split_summary = build_train_val_datasets(config)
        logger.info(
            f"Dataset split: {split_summary['train']} train, {split_summary['val']} val"
        )
        if split_summary["test"] > 0:
            logger.info(f"Detected {split_summary['test']} explicit test pairs (unused during training)")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            persistent_workers=config.training.num_workers > 0,
            prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
            collate_fn=collate_fn,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

        # Build optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
        )
        self.optimizer.zero_grad(set_to_none=True)

        # Build scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=config.training.max_steps,
        )

        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp else None

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

        # Optional S3 checkpoint uploads
        self.s3_bucket = self.runtime.s3_bucket
        self.s3_prefix = self.runtime.s3_checkpoint_prefix
        self.s3_client = None
        if self.runtime.enable_s3:
            if not self.s3_bucket:
                raise RuntimeError("INSTR_ENABLE_S3=true but INSTR_S3_BUCKET is not set")
            if not S3_AVAILABLE:
                raise RuntimeError("boto3 not installed - required for checkpoint uploads. pip install boto3")
            try:
                self.s3_client = boto3.client("s3", region_name=self.runtime.aws_region)
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                test_key = f"{self.s3_prefix}/_write_test.txt"
                self.s3_client.put_object(Bucket=self.s3_bucket, Key=test_key, Body=b"test")
                self.s3_client.delete_object(Bucket=self.s3_bucket, Key=test_key)
                logger.info(f"S3 initialized: s3://{self.s3_bucket}/{self.s3_prefix}")
            except Exception as e:
                raise RuntimeError(f"S3 access failed for optional checkpoint uploads: {e}")
        else:
            logger.info("S3 checkpoint uploads disabled")

        # Initialize W&B
        self.wandb_run = None
        self._init_wandb()

        # Load checkpoint if provided
        if config.checkpoint_path:
            self.load_checkpoint(config.checkpoint_path)

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not self.runtime.enable_wandb:
            logger.info("W&B disabled")
            return

        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed, logging disabled. pip install wandb")
            return

        try:
            # Login with API key if provided
            api_key = self.runtime.wandb_api_key
            if api_key:
                wandb.login(key=api_key, relogin=True)

            self.wandb_run = wandb.init(
                project=self.runtime.wandb_project or "ml",
                entity=self.runtime.wandb_entity,
                name=self.config.experiment_name,
                config={
                    "audio": vars(self.config.audio) if hasattr(self.config.audio, '__dict__') else str(self.config.audio),
                    "model": vars(self.config.model) if hasattr(self.config.model, '__dict__') else str(self.config.model),
                    "training": vars(self.config.training) if hasattr(self.config.training, '__dict__') else str(self.config.training),
                    "loss": vars(self.config.loss) if hasattr(self.config.loss, '__dict__') else str(self.config.loss),
                },
                resume="allow",
            )
            logger.info(f"W&B initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")
            self.wandb_run = None

    def _log_wandb(self, metrics: dict, step: int):
        """Log metrics to W&B."""
        if self.wandb_run:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B log failed: {e}")

    def train(self):
        """Main training loop with tqdm progress bar and W&B logging."""
        self.model.train()
        config = self.config.training

        accumulation_steps = config.gradient_accumulation
        log_losses: Dict[str, float] = {}
        accumulated_steps = 0

        logger.info(f"Starting training from step {self.global_step}")
        logger.info(f"Training for {config.max_steps} steps")
        effective_batch = config.batch_size * accumulation_steps
        logger.info(
            f"Effective batch size: {effective_batch} "
            f"(batch={config.batch_size} x accum={accumulation_steps})"
        )

        pbar = tqdm(total=config.max_steps, initial=self.global_step, desc="Training", ascii=True, dynamic_ncols=True)

        data_iter = iter(self.train_loader)

        while self.global_step < config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Move to device
            original_L = batch["original_L"].to(self.device, non_blocking=self.pin_memory)
            original_R = batch["original_R"].to(self.device, non_blocking=self.pin_memory)
            target_L = batch["inst_L"].to(self.device, non_blocking=self.pin_memory)
            target_R = batch["inst_R"].to(self.device, non_blocking=self.pin_memory)

            # Forward pass with optional AMP
            with autocast(device_type=self.device.type, enabled=config.use_amp):
                pred_L, pred_R = self.model(original_L, original_R)
                loss, loss_dict = self.loss_fn(pred_L, pred_R, target_L, target_R)
                loss = loss / accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate losses for logging (will average when logging)
            for k, v in loss_dict.items():
                log_losses[k] = log_losses.get(k, 0) + v
            log_losses["_count"] = log_losses.get("_count", 0) + 1

            accumulated_steps += 1

            # Optimizer step
            if accumulated_steps >= accumulation_steps:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.max_grad_norm,
                )

                # Step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1
                accumulated_steps = 0
                pbar.update(1)

                # Update progress bar
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(build_progress_postfix(log_losses, lr))

                # Logging
                if self.global_step % config.log_every_steps == 0:
                    metrics = build_train_metrics(log_losses, lr)
                    self._log_wandb(metrics, self.global_step)
                    log_losses = {}

                if self.global_step % config.save_every_steps == 0:
                    self.save_checkpoint()

                if self.global_step % 100 == 0:
                    try:
                        self.generate_sample(batch)
                    except Exception as e:
                        tqdm.write(f"Failed to generate sample: {e}")

                if self.global_step % config.eval_every_steps == 0:
                    eval_metrics = self.evaluate()
                    eval_loss = eval_metrics["loss"]
                    eval_sdr = eval_metrics["sdr"]
                    eval_min = eval_metrics["sdr_min"]
                    eval_max = eval_metrics["sdr_max"]
                    eval_si_sdr = eval_metrics["si_sdr"]
                    self._log_wandb({
                        "eval/loss": eval_loss,
                        "eval/sdr": eval_sdr,
                        "eval/sdr_min": eval_min,
                        "eval/sdr_max": eval_max,
                        "eval/si_sdr": eval_si_sdr,
                        "eval/n_batches": eval_metrics["n_batches"],
                    }, self.global_step)
                    tqdm.write(f"Step {self.global_step} | Eval Loss: {eval_loss:.4f} | SDR: {eval_sdr:.2f} | SI-SDR: {eval_si_sdr:.2f} [{eval_min:.1f}, {eval_max:.1f}]")

                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.save_checkpoint(best=True)
                        tqdm.write(f"New best model saved!")

                    self.model.train()

        pbar.close()

        logger.info("Training complete!")
        self.save_checkpoint()
        if self.wandb_run:
            wandb.finish()

    @torch.no_grad()
    def _compute_sdr(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Signal-to-Distortion Ratio (SDR) in dB.

        Standard SDR metric used for source separation evaluation.
        SDR = 10 * log10(||target||^2 / ||target - pred||^2)

        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
        Returns:
            sdr: (B,) SDR in dB for each sample (filtered for valid chunks)
        """
        # Ensure float32
        pred = pred.float()
        target = target.float()

        # Skip near-silent chunks (unreliable SDR)
        target_rms = (target ** 2).mean(dim=-1).sqrt()
        valid_mask = target_rms > 0.01  # Only chunks with meaningful audio

        # Compute powers
        target_power = (target ** 2).sum(dim=-1) + eps
        noise_power = ((target - pred) ** 2).sum(dim=-1) + eps

        sdr = 10 * torch.log10(target_power / noise_power)

        # Filter out invalid chunks and clamp extreme values
        sdr = torch.where(valid_mask, sdr, torch.zeros_like(sdr))
        sdr = torch.clamp(sdr, min=-10, max=30)  # Reasonable SDR range

        # Return only valid SDRs (or zeros if none valid)
        if valid_mask.any():
            return sdr[valid_mask]
        return sdr

    @torch.no_grad()
    def _compute_si_sdr(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute Scale-Invariant SDR (SI-SDR) in dB.

        SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        where s_target = (<pred, target> / ||target||^2) * target

        Args:
            pred: (B, T) predicted waveform
            target: (B, T) target waveform
        Returns:
            si_sdr: (B,) SI-SDR in dB for each sample
        """
        pred = pred.float()
        target = target.float()

        # Skip near-silent chunks
        target_rms = (target ** 2).mean(dim=-1).sqrt()
        valid_mask = target_rms > 0.01

        # Zero-mean
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        # SI-SDR
        dot = (pred * target).sum(dim=-1, keepdim=True)
        s_target_sq = (target ** 2).sum(dim=-1, keepdim=True) + eps
        s_target = dot * target / s_target_sq
        e_noise = pred - s_target

        si_sdr = 10 * torch.log10(
            (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
        )

        si_sdr = torch.where(valid_mask, si_sdr, torch.zeros_like(si_sdr))
        si_sdr = torch.clamp(si_sdr, min=-10, max=30)

        if valid_mask.any():
            return si_sdr[valid_mask]
        return si_sdr

    def evaluate(self, max_batches: Optional[int] = None) -> dict:
        """
        Evaluate on validation set, returns loss and SDR metrics.

        Args:
            max_batches: Max batches to eval. None = eval entire val set.
        """
        self.model.eval()

        total_loss = 0.0
        all_sdrs = []
        all_si_sdrs = []
        count = 0
        sample_batch = None

        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break

            if sample_batch is None:
                sample_batch = batch

            original_L = batch["original_L"].to(self.device, non_blocking=self.pin_memory)
            original_R = batch["original_R"].to(self.device, non_blocking=self.pin_memory)
            target_L = batch["inst_L"].to(self.device, non_blocking=self.pin_memory)
            target_R = batch["inst_R"].to(self.device, non_blocking=self.pin_memory)

            with torch.no_grad():
                with autocast(device_type=self.device.type, enabled=self.config.training.use_amp):
                    pred_L, pred_R = self.model(original_L, original_R)
                    loss, _ = self.loss_fn(pred_L, pred_R, target_L, target_R)

                # Compute both SDR and SI-SDR
                sdr_L = self._compute_sdr(pred_L.float(), target_L.float())
                sdr_R = self._compute_sdr(pred_R.float(), target_R.float())
                si_sdr_L = self._compute_si_sdr(pred_L.float(), target_L.float())
                si_sdr_R = self._compute_si_sdr(pred_R.float(), target_R.float())

                if len(sdr_L) > 0:
                    all_sdrs.extend(sdr_L.cpu().tolist())
                if len(sdr_R) > 0:
                    all_sdrs.extend(sdr_R.cpu().tolist())
                if len(si_sdr_L) > 0:
                    all_si_sdrs.extend(si_sdr_L.cpu().tolist())
                if len(si_sdr_R) > 0:
                    all_si_sdrs.extend(si_sdr_R.cpu().tolist())

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / max(count, 1)
        median_sdr = np.median(all_sdrs) if all_sdrs else 0.0
        min_sdr = np.min(all_sdrs) if all_sdrs else 0.0
        max_sdr = np.max(all_sdrs) if all_sdrs else 0.0
        median_si_sdr = np.median(all_si_sdrs) if all_si_sdrs else 0.0

        metrics = {
            "loss": avg_loss,
            "sdr": median_sdr,
            "sdr_min": min_sdr,
            "sdr_max": max_sdr,
            "si_sdr": median_si_sdr,
            "n_batches": count,
        }

        # Generate and upload eval samples
        if sample_batch is not None:
            self._generate_eval_sample(sample_batch, metrics)

        return metrics

    @torch.no_grad()
    def _generate_eval_sample(self, batch: dict, metrics: dict):
        """Generate eval audio samples and upload to S3."""
        if not TORCHAUDIO_AVAILABLE:
            return

        try:
            # Take first sample from batch
            original_L = batch["original_L"][0:1].to(self.device, non_blocking=self.pin_memory)
            original_R = batch["original_R"][0:1].to(self.device, non_blocking=self.pin_memory)
            target_L = batch["inst_L"][0:1].to(self.device, non_blocking=self.pin_memory)
            target_R = batch["inst_R"][0:1].to(self.device, non_blocking=self.pin_memory)

            # Forward pass
            with autocast(device_type=self.device.type, enabled=self.config.training.use_amp):
                pred_L, pred_R = self.model(original_L, original_R)

            sr = self.config.audio.sample_rate
            step = self.global_step

            # Stack stereo channels
            original = torch.stack([original_L[0], original_R[0]], dim=0).cpu()
            target = torch.stack([target_L[0], target_R[0]], dim=0).cpu()
            predicted = torch.stack([pred_L[0], pred_R[0]], dim=0).float().cpu()
            residual = original - predicted

            # Normalize to prevent clipping
            for audio in [original, target, predicted, residual]:
                max_val = audio.abs().max()
                if max_val > 1.0:
                    audio.div_(max_val)

            # Save eval samples locally
            eval_samples_dir = self.output_dir / "eval_samples"
            eval_samples_dir.mkdir(exist_ok=True)

            paths = {
                "input": eval_samples_dir / f"eval_{step:06d}_input.wav",
                "target": eval_samples_dir / f"eval_{step:06d}_target.wav",
                "predicted": eval_samples_dir / f"eval_{step:06d}_predicted.wav",
                "residual": eval_samples_dir / f"eval_{step:06d}_residual.wav",
            }

            torchaudio.save(str(paths["input"]), original, sr)
            torchaudio.save(str(paths["target"]), target, sr)
            torchaudio.save(str(paths["predicted"]), predicted, sr)
            torchaudio.save(str(paths["residual"]), residual, sr)

            # Save metrics JSON
            metrics_path = eval_samples_dir / f"eval_{step:06d}_metrics.json"
            import json
            with open(metrics_path, "w") as f:
                json.dump({
                    "step": step,
                    "loss": metrics["loss"],
                    "sdr": metrics["sdr"],
                    "sdr_min": metrics["sdr_min"],
                    "sdr_max": metrics["sdr_max"],
                    "si_sdr": metrics["si_sdr"],
                    "n_batches": metrics["n_batches"],
                }, f, indent=2)

            tqdm.write(f"Generated eval samples at step {step} (SDR: {metrics['sdr']:.2f} | SI-SDR: {metrics['si_sdr']:.2f})")

            # Upload to S3
            if self.s3_client:
                s3_base = f"{self.s3_prefix}/{self.config.experiment_name}/eval"
                for name, path in paths.items():
                    try:
                        s3_key = f"{s3_base}/step_{step:06d}_{name}.wav"
                        self.s3_client.upload_file(str(path), self.s3_bucket, s3_key)
                    except Exception:
                        pass
                # Upload metrics
                try:
                    s3_key = f"{s3_base}/step_{step:06d}_metrics.json"
                    self.s3_client.upload_file(str(metrics_path), self.s3_bucket, s3_key)
                    tqdm.write(f"Uploaded eval to S3: s3://{self.s3_bucket}/{s3_base}/step_{step:06d}_*")
                except Exception as e:
                    tqdm.write(f"S3 eval upload failed: {e}")

            # Log to W&B
            if self.wandb_run and WANDB_AVAILABLE:
                try:
                    wandb.log({
                        "eval_samples/input": wandb.Audio(original.numpy().T, sample_rate=sr, caption="Eval Input"),
                        "eval_samples/target": wandb.Audio(target.numpy().T, sample_rate=sr, caption="Eval Target"),
                        "eval_samples/predicted": wandb.Audio(predicted.numpy().T, sample_rate=sr, caption="Eval Predicted"),
                        "eval_samples/residual": wandb.Audio(residual.numpy().T, sample_rate=sr, caption="Eval Residual"),
                    }, step=self.global_step)
                except Exception as e:
                    tqdm.write(f"W&B eval audio log failed: {e}")

        except Exception as e:
            tqdm.write(f"Eval sample generation failed: {e}")

    @torch.no_grad()
    def generate_sample(self, batch: dict):
        """Generate audio samples from current batch and log to W&B."""
        if not TORCHAUDIO_AVAILABLE:
            return

        self.model.eval()

        try:
            # Take first sample from batch
            original_L = batch["original_L"][0:1].to(self.device, non_blocking=self.pin_memory)
            original_R = batch["original_R"][0:1].to(self.device, non_blocking=self.pin_memory)
            target_L = batch["inst_L"][0:1].to(self.device, non_blocking=self.pin_memory)
            target_R = batch["inst_R"][0:1].to(self.device, non_blocking=self.pin_memory)

            # Forward pass
            with autocast(device_type=self.device.type, enabled=self.config.training.use_amp):
                pred_L, pred_R = self.model(original_L, original_R)

            # Convert to numpy for saving
            sr = self.config.audio.sample_rate

            # Stack stereo channels: (1, T) -> (2, T)
            original = torch.stack([original_L[0], original_R[0]], dim=0).cpu()
            target = torch.stack([target_L[0], target_R[0]], dim=0).cpu()
            predicted = torch.stack([pred_L[0], pred_R[0]], dim=0).float().cpu()
            residual = original - predicted  # vocals = original - instrumental

            # Normalize to prevent clipping
            for audio in [original, target, predicted, residual]:
                max_val = audio.abs().max()
                if max_val > 1.0:
                    audio.div_(max_val)

            # Save locally
            step = self.global_step
            paths = {
                "input": self.samples_dir / f"step_{step:06d}_input.wav",
                "target": self.samples_dir / f"step_{step:06d}_target.wav",
                "predicted": self.samples_dir / f"step_{step:06d}_predicted.wav",
                "residual": self.samples_dir / f"step_{step:06d}_residual.wav",
            }

            torchaudio.save(str(paths["input"]), original, sr)
            torchaudio.save(str(paths["target"]), target, sr)
            torchaudio.save(str(paths["predicted"]), predicted, sr)
            torchaudio.save(str(paths["residual"]), residual, sr)

            tqdm.write(f"Generated samples at step {step}")

            # Log to W&B (pass numpy arrays directly for better compatibility)
            if self.wandb_run and WANDB_AVAILABLE:
                try:
                    # wandb.Audio expects (samples, channels) or (samples,) numpy array
                    # Our tensors are (2, T), need to transpose to (T, 2)
                    wandb.log({
                        "samples/input": wandb.Audio(original.numpy().T, sample_rate=sr, caption="Input (with vocals)"),
                        "samples/target": wandb.Audio(target.numpy().T, sample_rate=sr, caption="Target (instrumental)"),
                        "samples/predicted": wandb.Audio(predicted.numpy().T, sample_rate=sr, caption="Predicted (instrumental)"),
                        "samples/residual": wandb.Audio(residual.numpy().T, sample_rate=sr, caption="Residual (vocals)"),
                    }, step=self.global_step)
                    tqdm.write(f"Logged audio samples to W&B at step {step}")
                except Exception as e:
                    tqdm.write(f"Failed to log audio to W&B: {e}")
            else:
                tqdm.write(f"W&B not available for audio logging (run={self.wandb_run}, available={WANDB_AVAILABLE})")

            # Upload to S3
            if self.s3_client:
                for name, path in paths.items():
                    try:
                        s3_key = f"{self.s3_prefix}/{self.config.experiment_name}/samples/step_{step:06d}_{name}.wav"
                        self.s3_client.upload_file(str(path), self.s3_bucket, s3_key)
                    except Exception:
                        pass

        except Exception as e:
            tqdm.write(f"Sample generation failed: {e}")

        self.model.train()

    def save_checkpoint(self, best: bool = False):
        """Save model checkpoint locally and to S3."""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model_unwrapped.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": {
                "n_fft": self.config.audio.n_fft,
                "hop_length": self.config.audio.hop_length,
                "sample_rate": self.config.audio.sample_rate,
                "chunk_seconds": self.config.audio.chunk_seconds,
                "d_model": self.config.model.d_model,
                "n_heads": self.config.model.n_heads,
                "n_encoder_layers": self.config.model.n_encoder_layers,
                "n_decoder_layers": self.config.model.n_decoder_layers,
                "n_bottleneck_layers": self.config.model.n_bottleneck_layers,
                "d_state": getattr(self.config.model, 'd_state', 32),
                "use_mid_side": getattr(self.config.model, 'use_mid_side', False),
            },
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save regular checkpoint
        path = self.output_dir / f"step_{self.global_step:06d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Upload to S3
        if self.s3_client:
            try:
                s3_key = f"{self.s3_prefix}/{self.config.experiment_name}/step_{self.global_step:06d}.pt"
                self.s3_client.upload_file(str(path), self.s3_bucket, s3_key)
                logger.info(f"Uploaded to S3: s3://{self.s3_bucket}/{s3_key}")
            except Exception as e:
                logger.warning(f"S3 upload failed: {e}")

        # Save best checkpoint
        if best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            if self.s3_client:
                try:
                    s3_key = f"{self.s3_prefix}/{self.config.experiment_name}/best.pt"
                    self.s3_client.upload_file(str(best_path), self.s3_bucket, s3_key)
                except Exception:
                    pass

        # Also save latest
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        if self.s3_client:
            try:
                s3_key = f"{self.s3_prefix}/{self.config.experiment_name}/latest.pt"
                self.s3_client.upload_file(str(latest_path), self.s3_bucket, s3_key)
            except Exception:
                pass

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {path}")

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model_unwrapped.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed from step {self.global_step}")


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train vocal separator")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--data_dirs", type=str, help="Data directories (comma-delimited)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--checkpoint", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    runtime = load_runtime_config()

    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = base_config()

    config.apply_runtime_defaults(runtime)

    # Override with CLI args only if explicitly provided
    if args.data_dirs:
        config.data_dirs = args.data_dirs
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint

    trainer = Trainer(config, runtime=runtime)
    trainer.train()


if __name__ == "__main__":
    main()
