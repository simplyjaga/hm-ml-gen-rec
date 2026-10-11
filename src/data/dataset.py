from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.config import GenRecConfig


class HMSequenceDataset(Dataset):
    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        config: GenRecConfig
    ) -> None:
        self.data: pd.DataFrame = pd.read_parquet(parquet_path)
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = config.max_seq_length

        # Enforce left padding for causal sequence encoding
        self.tokenizer.padding_side = "left"

        # FIX: Enforce left truncation so we never lose the final "Next likely purchase:" prompt
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build target vocabulary for the custom projection head
        self.unique_targets: list[str] = self.data['target_id'].unique().tolist()
        self.target2idx: dict[str, int] = {
            target_id: idx for idx, target_id in enumerate(self.unique_targets)
        }

        # Expose num_items so the model knows how big to make the Embedding layer
        self.num_items: int = len(self.unique_targets)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]

        prompt: str = (
            f"Customer Profile: {row['user_profile']}\n"
            f"Purchase History:\n{row['context_texts']}\n\n"
            f"Next likely purchase:"
        )

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        target_idx: int = self.target2idx[row['target_id']]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_idx": torch.tensor(target_idx, dtype=torch.long)
        }
