# src/utils.py
import torch

from src.configs.model_config import GenRecConfig


def extract_trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Extracts only the parameters that require gradients (LoRA, heads, embeddings).
    Ignores the frozen LLM backbone to save disk space and memory.
    """
    trainable_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_state_dict[name] = param.cpu().detach()
    return trainable_state_dict

def override_config_from_wandb(base_config: GenRecConfig, wandb_config) -> GenRecConfig:
    """
    Dynamically overrides the dataclass configuration with W&B sweep parameters.
    Prevents massive if/else chains.
    """
    if not wandb_config:
        return base_config

    for key, value in wandb_config.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)

    return base_config
