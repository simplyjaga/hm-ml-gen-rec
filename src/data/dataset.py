from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.configs.model_config import GenRecConfig


class HMSequenceDataset(Dataset):
    def __init__(self, parquet_path: str | Path, tokenizer: PreTrainedTokenizerBase,
                 config: GenRecConfig, target2idx: dict[str, int] | None = None,
                 data: pd.DataFrame | None = None) -> None:
        self.data = data.copy() if data is not None else pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = config.max_seq_length
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # The caller can provide a vocabulary derived exclusively from train targets.
        self.target2idx = target2idx or {
            str(item): idx for idx, item in enumerate(self.data["target_id"].astype(str).unique())
        }
        self.num_items = len(self.target2idx)
        self.data = self.data[self.data["target_id"].astype(str).isin(self.target2idx)].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        prompt = (f"Customer Profile: {row['user_profile']}\n"
                  f"Purchase History:\n{row['context_texts']}\n\nNext likely purchase:")
        encoded = self.tokenizer(prompt, truncation=True, max_length=self.max_length,
                                 padding="max_length", return_tensors="pt")
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "target_idx": torch.tensor(self.target2idx[str(row["target_id"])], dtype=torch.long),
        }
