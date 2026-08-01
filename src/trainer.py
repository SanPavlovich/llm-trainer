from pathlib import Path
import contextlib
import functools
import logging
from tqdm.auto import tqdm
from typing import Any
import torch
from torch import Tensor
from torch.profiler import profile as Profiler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor

from src.schemas import TrainerConfig, ProfilerConfig
from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer
from src.utils import cross_entropy_loss, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


def trace_handler(prof: Profiler, traces_dir: Path) -> None:
    prof.export_chrome_trace(
        str(traces_dir / f"trace.json")
    )


class Trainer:
    def __init__(
        self,
        train_config: TrainerConfig,
        tokenizer: ByteLevelBPETokenizer,
        model: Any,
        tensorboard_dir: Path,
        checkpoint_dir: Path,
        profiler_dir: Path | None = None,
        profiler_config: ProfilerConfig | None = None,
        optimizer: torch.optim.Optimizer|None=None,
        scheduler: torch.optim.lr_scheduler.LRScheduler|None=None,
        valid_texts: list[str] | None = None,
        valid_images: dict | None = None,
    ) -> None:
        self.train_config = train_config
        self.checkpoint_dir = checkpoint_dir
        self.profiler_dir = profiler_dir
        self.profiler_config = profiler_config
        self.tokenizer = tokenizer
        self.model = model
        self.valid_texts = valid_texts
        self.valid_images = valid_images
            
        if optimizer is None:
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.train_config.learning_rate, 
                weight_decay=self.train_config.weight_decay
            )
        else:
            self.optimizer = optimizer
        if scheduler is None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=0.1 * self.train_config.n_steps, 
                num_training_steps=self.train_config.n_steps
            )
        else:
            self.scheduler = scheduler
        self.global_step = 0
        self.writer = SummaryWriter(tensorboard_dir / "train")
        self.valid_writer = SummaryWriter(tensorboard_dir / "valid")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        logger.info(f"running on device {self.device}")

        self._setup_amp()
        self._setup_compile()

    def _setup_compile(self) -> None:
        want_compile = bool(getattr(self.train_config, "compile", False))
        if want_compile and self.device == "cuda":
            mode = getattr(self.train_config, "compile_mode", "default")
            logger.info(f"torch.compile enabled (mode={mode}); first steps include compilation")
            self.train_model = torch.compile(self.model, mode=mode)
        else:
            if want_compile:
                logger.warning("compile requested but device is not CUDA; running eager")
            self.train_model = self.model

    def _setup_amp(self) -> None:
        """Configure mixed-precision (autocast) state from the trainer config.

        bf16 needs no GradScaler; fp16 does, so a scaler is created only then.
        AMP is disabled on CPU (no autocast benefit here) even if requested.
        """
        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        self.amp_dtype = dtypes.get(self.train_config.amp_dtype, torch.bfloat16)
        self.amp_enabled = bool(self.train_config.mixed_precision) and self.device == "cuda"

        # GradScaler is required for fp16 (to avoid gradient underflow); bf16 has
        # the same exponent range as fp32 and does not need it.
        use_scaler = self.amp_enabled and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        if self.amp_enabled:
            logger.info(f"mixed precision enabled: autocast dtype={self.amp_dtype}, "
                        f"grad_scaler={'on' if use_scaler else 'off'}")

    def _autocast(self):
        """Autocast context for the forward/loss; a no-op when AMP is disabled."""
        return torch.autocast(
            device_type=self.device if self.device != "mps" else "cpu",
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        )

    def _build_profiler(self) -> Profiler | None:
        """Create a torch.profiler that traces the configured iteration(s).

        Traces are written to profiler_dir in TensorBoard format (viewable via
        the PyTorch Profiler TensorBoard plugin) and as Chrome trace JSON.
        Returns None when profiling is disabled.
        """
        cfg = self.profiler_config
        if cfg is None or not cfg.enabled or self.profiler_dir is None:
            return None

        self.profiler_dir.mkdir(parents=True, exist_ok=True)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        return Profiler(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=cfg.wait, warmup=cfg.warmup, active=cfg.active, repeat=cfg.repeat
            ),
            on_trace_ready=functools.partial(trace_handler, traces_dir=self.profiler_dir),
            record_shapes=cfg.record_shapes,
            profile_memory=cfg.profile_memory,
            with_stack=cfg.with_stack,
        )

    def _memory_snapshot_enabled(self) -> bool:
        cfg = self.profiler_config
        return (
            cfg is not None
            and cfg.memory_snapshot
            and self.device == "cuda"
            and self.profiler_dir is not None
        )

    def _memory_snapshot_stop_step(self) -> int:
        cfg = self.profiler_config
        stop = cfg.memory_snapshot_stop_step
        return cfg.memory_snapshot_step if stop is None else stop

    def _maybe_start_memory_snapshot(self, iter_num: int) -> None:
        if not self._memory_snapshot_enabled():
            return
        if iter_num == self.profiler_config.memory_snapshot_step - 1:
            torch.cuda.memory._record_memory_history(
                max_entries=self.profiler_config.memory_snapshot_max_entries
            )

    def _maybe_dump_memory_snapshot(self, iter_num: int) -> None:
        """Dump and stop recording right after the last target iteration."""
        if not self._memory_snapshot_enabled():
            return
        start = self.profiler_config.memory_snapshot_step
        stop = self._memory_snapshot_stop_step()
        if iter_num == stop:
            self.profiler_dir.mkdir(parents=True, exist_ok=True)
            span = f"step{start}" if start == stop else f"steps{start}-{stop}"
            snapshot_path = self.profiler_dir / f"memory_snapshot_{span}.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)  # stop & free buffer
            logger.info(f"CUDA memory snapshot saved to {snapshot_path}")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tensor:
        self.model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with self._autocast():
                logits = self.train_model(**batch)
                loss_mask = None if "image_ids_mask" not in batch else ~batch["image_ids_mask"].bool()
                loss = cross_entropy_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    logits,
                    loss_mask=loss_mask
                )
            # accumulate in fp32 to avoid drift over many val batches
            val_loss += loss.float()

        return val_loss / len(val_loader)

    @torch.no_grad()
    def _log_grad_norm_per_layer(self) -> None:
        sq_by_group: dict[str, float] = {}
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            parts = name.split(".")
            if parts[0] == "layers" and len(parts) > 1:
                group = f"layer_{int(parts[1]):02d}"
            else:
                group = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
            sq_by_group[group] = sq_by_group.get(group, 0.0) + p.grad.detach().float().pow(2).sum().item()

        for group, sq in sq_by_group.items():
            self.writer.add_scalar(f"grad_norm_per_layer/{group}", sq ** 0.5, self.global_step)

    def save_checkpoint(self) -> None:
        checkpoint_name = f"{self.checkpoint_dir}/{self.global_step}.pkl"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.global_step,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
        }, checkpoint_name)
        logger.info(f"step {self.global_step}; checkpoint saved at {checkpoint_name}")
    
    def load_checkpoint(self,) -> None:
        pass
    
    def log_model_samples(self, text_field: str) -> None:
        if self.valid_images is not None:
            self._log_visual_samples(text_field)
        else:
            self._log_text_samples(text_field)

    @torch.no_grad()
    def _log_text_samples(self, text_field):
        result_texts = []
        for text in self.valid_texts:
            input_ids = torch.tensor(self.tokenizer.encode(text)[:-1], device=self.device)[None, :]
            model_output = self.model.generate(
                input_ids,
                max_new_tokens=200,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=10
            )
            out_text = self.tokenizer.decode(model_output[0].tolist())
            result_texts.append(out_text)

        for i, (start_text, res_text) in enumerate(zip(self.valid_texts, result_texts)):
            self.writer.add_text(
                text_field,
                f"**sample {i+1}:**\n\n[{res_text[:len(start_text)]}]{res_text[len(start_text):]}",
                self.global_step
            )

    @torch.no_grad()
    def _log_visual_samples(self, text_field: str) -> None:
        """Generate a caption for each fixed validation image and log both the
        image and the generated text to TensorBoard."""
        pixel_values = self.valid_images["pixel_values"]          # [K, 3, H, W]
        display_images = self.valid_images.get("display_images")  # [K, 3, H, W] or None
        captions = self.valid_images.get("captions")              # list[str] or None
        num_patches = self.model.adapter_config.num_image_patches

        # image prefix: [IMG_START] [IMG]*P [IMG_END]
        prefix_ids = (
            [self.tokenizer.image_start_token_id]
            + [self.tokenizer.image_token_id] * num_patches
            + [self.tokenizer.image_end_token_id]
        )
        prefix_mask = [False] + [True] * num_patches + [False]

        for i in range(pixel_values.shape[0]):
            px = pixel_values[i : i + 1].to(self.device, non_blocking=True)  # [1, 3, H, W]
            input_ids = torch.tensor(prefix_ids, device=self.device)[None, :]  # [1, L0]
            image_ids_mask = torch.tensor(prefix_mask, device=self.device)[None, :]

            out = self.model.generate(
                input_ids,
                pixel_values=px,
                image_ids_mask=image_ids_mask,
                max_new_tokens=64,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=10,
            )
            # Drop the image-prefix ids; decode only the generated text tail.
            gen_ids = out[0, len(prefix_ids):].tolist()
            gen_text = self.tokenizer.decode(gen_ids)

            ref = f"\n\n**reference:** {captions[i]}" if captions else ""
            self.writer.add_text(
                text_field,
                f"**image {i+1}:** \n\n**generated text:**{gen_text}\n\n**reference text:**{ref}",
                self.global_step,
            )
            if display_images is not None:
                self.writer.add_image(
                    f"{text_field}/image_{i+1}",
                    display_images[i],
                    self.global_step,
                )

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader | None,
    ) -> None:
        self.model.to(self.device)
        self.model.train()

        profiler = self._build_profiler()
        data_iter = iter(train_loader)
        self.train_loss, self.valid_loss = None, None

        with profiler if profiler is not None else contextlib.nullcontext():
            for iter_num in tqdm(range(self.train_config.n_steps), desc="train steps"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    batch = next(data_iter)

                self._maybe_start_memory_snapshot(iter_num)

                with torch.profiler.record_function("train_step"):
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    # Forward + loss run under autocast (bf16/fp16); precision-
                    # sensitive ops (softmax, cross-entropy, RMSNorm, RoPE) stay fp32.
                    with self._autocast():
                        with torch.profiler.record_function("model_forward"):
                            logits = self.train_model(**batch)  # [bs; seq len; vocab size]
                        with torch.profiler.record_function("cross_entropy_loss"):
                            # For the VLM, skip [IMG] placeholder targets in the loss:
                            # they aren't real language, so predicting them is meaningless.
                            loss_mask = None
                            if "image_ids_mask" in batch:
                                loss_mask = ~batch["image_ids_mask"].bool()
                            loss = cross_entropy_loss(
                                batch["input_ids"], batch["attention_mask"], logits, loss_mask=loss_mask
                            )
                    with torch.profiler.record_function("loss_backward"):
                        # scaler is a no-op unless fp16 AMP is on.
                        self.scaler.scale(loss).backward()
                    with torch.profiler.record_function("clip_grad_norm"):
                        # Unscale before clipping so the norm/threshold are in true units.
                        self.scaler.unscale_(self.optimizer)
                        every = self.train_config.log_grad_norm_per_layer_every
                        if every and self.global_step % every == 0:
                            self._log_grad_norm_per_layer()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.clip_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.train_loss = loss.item()
                self.writer.add_scalar("loss", self.train_loss, self.global_step)
                self.writer.add_scalar("grad_norm", grad_norm, self.global_step)
                self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], self.global_step)

                if (
                    val_loader is not None
                    and iter_num > 0
                    and self.train_config.val_every_n_steps != 0
                    and iter_num % self.train_config.val_every_n_steps == 0
                ):
                    val_loss = self.validate(val_loader)
                    self.valid_loss = val_loss.item()
                    self.valid_writer.add_scalar("loss", val_loss, self.global_step)
                    self.log_model_samples(text_field=f"model samples step {self.global_step}")
                    self.model.train()

                    if self.train_config.save_checkpoint:
                        self.save_checkpoint()

                self._maybe_dump_memory_snapshot(iter_num)

                # advance the profiler schedule (no-op when profiling disabled)
                if profiler is not None:
                    profiler.step()

                self.global_step += 1

        if self.train_config.val_after_train and val_loader is not None:
            val_loss = self.validate(val_loader)
            self.valid_loss = val_loss
            self.valid_writer.add_scalar("loss", val_loss, self.global_step)
            self.log_model_samples(text_field="model samples train end")
        
        if self.train_config.save_checkpoint:
            self.save_checkpoint()