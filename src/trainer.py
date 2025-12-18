from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from src.schemas import TrainerConfig
from src.tokenizer import ByteLevelBPETokenizer
from src.model import TransformerForCausalLM
from src.utils import cross_entropy_loss, get_linear_schedule_with_warmup


class Trainer:
    def __init__(
        self,
        train_config: TrainerConfig,
        tokenizer: ByteLevelBPETokenizer,
        model: TransformerForCausalLM,
        valid_texts: list[str],
        log_dir: Path,
        optimizer: torch.optim.Optimizer|None=None,
        scheduler: torch.optim.lr_scheduler.LRScheduler|None=None,
    ) -> None:
        self.train_config = train_config
        self.log_dir = log_dir
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
        self.writer = SummaryWriter(log_dir / "train")
        self.valid_writer = SummaryWriter(log_dir / "valid")

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print("running on device", self.device)

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True).long()
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = self.model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            val_loss += cross_entropy_loss(input_ids, attention_mask, logits)
        return val_loss / len(val_loader)
    
    def save_checkpoint(self):
        checkpoint_name = f"{self.log_dir}/{self.global_step}.pkl"
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


    def train(self, train_loader, val_loader, save_checkpoint=True):
        self.model.to(self.device)
        self.model.train()

        data_iter = iter(train_loader)
        for iter_num in tqdm(range(self.train_config.n_steps), desc="train steps"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids, attention_mask = batch
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)

            logits = self.model(input_ids, attention_mask)  # [bs; seq len; vocab size]
            loss = cross_entropy_loss(input_ids, attention_mask, logits)

            # backprop and update the parameters
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            self.train_loss = loss.item()
            self.writer.add_scalar("loss", self.train_loss, self.global_step)
            self.writer.add_scalar("grad_norm", grad_norm, self.global_step)
            self.writer.add_scalar("learning_rate", self.scheduler.get_last_lr()[0], self.global_step)

            if iter_num > 0 and iter_num % self.train_config.val_every_n_steps == 0:
                val_loss = self.validate(val_loader)
                self.valid_loss = val_loss
                self.valid_writer.add_scalar("loss", val_loss, self.global_step)
                self.log_model_samples(text_field=f"model samples step {self.global_step}")
                self.model.train()

                if save_checkpoint:
                    self.save_checkpoint()
            
            self.global_step += 1

        val_loss = self.validate(val_loader)
        self.valid_loss = val_loss
        self.valid_writer.add_scalar("loss", val_loss, self.global_step)
        self.log_model_samples(text_field="model samples train end")
        
        if save_checkpoint:
            self.save_checkpoint()