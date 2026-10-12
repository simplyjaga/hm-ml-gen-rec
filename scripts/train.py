# scripts/train.py
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import wandb
from src.config import GenRecConfig
from src.data.dataset import HMSequenceDataset
from src.model.genrec import HMGenRecModel


def train(args):
    # 1. Setup Configuration & W&B
    config = GenRecConfig()
    if args.is_kaggle:
        config.batch_size = 16  # Higher batch size for Kaggle GPUs
        config.epochs = 5

    wandb.init(
        project="hm_genrec",
        config=config.to_wandb_dict(),
        mode="online"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    dataset_path = args.data_path if not args.dry_run else "data/processed/train_sample.parquet"
    dataset = HMSequenceDataset(dataset_path, tokenizer, config)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = HMGenRecModel(config, num_items=dataset.num_items).to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 5. Training Loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_idx = batch["target_idx"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, target_idx=target_idx)
            loss = outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train/batch_loss": loss.item()})

            if args.dry_run and batch_idx >= 5:
                print("Dry run completed 5 batches. Exiting successfully.")
                return

        epoch_avg_loss = epoch_loss / len(dataloader)
        wandb.log({"train/epoch_loss": epoch_avg_loss, "epoch": epoch})
        print(f"Epoch {epoch+1} Complete. Avg Loss: {epoch_avg_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/train_sample.parquet")
    parser.add_argument("--dry_run", action="store_true", help="Run 5 batches locally to verify code")
    parser.add_argument("--is_kaggle", action="store_true", help="Overrides config for Kaggle T4x2 environment")
    args = parser.parse_args()

    train(args)
