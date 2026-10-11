from dataclasses import asdict, dataclass


@dataclass
class GenRecConfig:
    # Model & Tokenizer
    backbone_name: str = "Qwen/Qwen2.5-0.5B"
    max_seq_length: int = 256

    # LoRA / PEFT Parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training Parameters
    batch_size: int = 4
    learning_rate: float = 2e-4
    epochs: int = 3

    def to_wandb_dict(self) -> dict:
        """Converts config to a dictionary for wandb.init()"""
        return asdict(self)
