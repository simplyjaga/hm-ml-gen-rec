# src/model/genrec.py
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModel

from src.configs.model_config import GenRecConfig


class HMGenRecModel(nn.Module):
    def __init__(self, config: GenRecConfig, num_items: int):
        super().__init__()
        self.config = config
        self.num_items = num_items

        # 1. Load Base Model (Raw Hidden States Only)
        # Using trust_remote_code=True may be required for some Qwen models
        base_model = AutoModel.from_pretrained(
            config.backbone_name,
            trust_remote_code=True,
            dtype=torch.bfloat16 # Use bfloat16 to save memory
        )

        # 2. Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.target_modules)
        )
        self.encoder = get_peft_model(base_model, peft_config)
        self.encoder.print_trainable_parameters()

        # 3. Custom Projection Head
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16)

        # 4. Item Catalog Embeddings
        self.item_embeddings = nn.Embedding(num_items, hidden_size, dtype=torch.bfloat16)

        # 5. Loss Function (Binary Cross Entropy with Logits)
        # We treat recommendation as multi-class classification over the catalog
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_idx: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Shape: (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.last_hidden_state

        # Because we use LEFT padding, the final token is always at index -1
        # Shape becomes: (batch_size, hidden_size)
        h_last = last_hidden_states[:, -1, :]

        # 3. Project to latent space
        user_vectors = self.projection(h_last)

        # 4. Compute scores against ALL items in the catalog
        # user_vectors shape: (batch_size, hidden_size)
        # item_embeddings.weight shape: (num_items, hidden_size)
        all_item_vectors = self.item_embeddings.weight

        # Matrix multiplication yields (batch_size, num_items) scores
        logits = torch.matmul(user_vectors, all_item_vectors.T)

        # Only calculate loss if targets are provided (Training/Validation)
        if target_idx is not None:
            loss = self.loss_fn(logits, target_idx)
        else:
            loss = None

        return {"logits": logits, "loss": loss}
