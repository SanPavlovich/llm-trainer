from pathlib import Path
import contextlib
import functools
from tqdm.auto import tqdm
import torch
from torch import Tensor
from torch.profiler import profile as Profiler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.schemas import TrainerConfig, ProfilerConfig
from src.tokenizer.bpe_tokenizer import ByteLevelBPETokenizer
from src.model import TransformerForCausalLM
from src.utils import cross_entropy_loss, get_linear_schedule_with_warmup


def trace_handler(prof: Profiler, traces_dir: Path) -> None:
    prof.export_chrome_trace(
        str(traces_dir / f"trace.json")
    )


class Trainer:
    def __init__(
        self,
        train_config: TrainerConfig,
        tokenizer: ByteLevelBPETokenizer,
        model: TransformerForCausalLM,
        valid_texts: list[str],
        tensorboard_dir: Path,
        checkpoint_dir: Path,
        profiler_dir: Path | None = None,
        profiler_config: ProfilerConfig | None = None,
        optimizer: torch.optim.Optimizer|None=None,
        scheduler: torch.optim.lr_scheduler.LRScheduler|None=None,
    ) -> None:
        self.train_config = train_config
        self.checkpoint_dir = checkpoint_dir
        self.profiler_dir = profiler_dir
        self.profiler_config = profiler_config
        self.valid_texts = valid_texts
        self.tokenizer = tokenizer
        self.model = model
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
        print("running on device", self.device)

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

    def _maybe_start_memory_snapshot(self, iter_num: int) -> None:
        """Start recording CUDA allocation history one iteration before the target,
        so the snapshot captures exactly the target iteration's allocations."""
        if not self._memory_snapshot_enabled():
            return
        if iter_num == self.profiler_config.memory_snapshot_step - 1:
            torch.cuda.memory._record_memory_history(
                max_entries=self.profiler_config.memory_snapshot_max_entries
            )

    def _maybe_dump_memory_snapshot(self, iter_num: int) -> None:
        """Dump and stop recording right after the target iteration."""
        if not self._memory_snapshot_enabled():
            return
        if iter_num == self.profiler_config.memory_snapshot_step:
            self.profiler_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = self.profiler_dir / f"memory_snapshot_step{iter_num}.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            torch.cuda.memory._record_memory_history(enabled=None)  # stop & free buffer
            print(f"CUDA memory snapshot saved to {snapshot_path}")

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tensor:
        self.model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True).long()
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = self.model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            val_loss += cross_entropy_loss(input_ids, attention_mask, logits)
        return val_loss / len(val_loader)
    
    def save_checkpoint(self) -> None:
        checkpoint_name = f"{self.checkpoint_dir}/{self.global_step}.pkl"
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.global_step,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
        }, checkpoint_name)
    
    def load_checkpoint(self,) -> None:
        pass
    
    def log_model_samples(self, text_field: str) -> None:
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


    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader | None, 
        save_checkpoint=True
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
                    input_ids, attention_mask = batch
                    input_ids = input_ids.to(self.device, non_blocking=True)
                    attention_mask = attention_mask.to(self.device, non_blocking=True)

                    with torch.profiler.record_function("model_forward"):
                        logits = self.model(input_ids, attention_mask)  # [bs; seq len; vocab size]
                    with torch.profiler.record_function("cross_entropy_loss"):
                        loss = cross_entropy_loss(input_ids, attention_mask, logits)
                    with torch.profiler.record_function("loss_backward"):
                        loss.backward()
                    with torch.profiler.record_function("clip_grad_norm"):
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.clip_grad_norm)
                    self.optimizer.step()
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

                    if save_checkpoint:
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
        
        if save_checkpoint:
            self.save_checkpoint()