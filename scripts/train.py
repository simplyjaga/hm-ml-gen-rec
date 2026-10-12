# scripts/train.py
import argparse
import os

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import extract_trainable_state_dict, override_config_from_wandb

import wandb
from src.configs.model_config import GenRecConfig
from src.data.dataset import HMSequenceDataset
from src.model.genrec import HMGenRecModel
from src.training.metrics import calculate_map_at_k


def train(cli_args=None):
    # 1. Initialize W&B.
    # If launched by a W&B sweep agent via the YAML file, this automatically
    # grabs the sweep ID and injects the hyperparameter choices into wandb.config.
    wandb.init(project="hm_genrec", mode="online")

    # 2. Setup Configuration
    # Base config -> override with Known CLI args -> override with W&B sweep parameters
    base_config = GenRecConfig()

    if cli_args is not None:
        if cli_args.epochs is not None:
            base_config.epochs = cli_args.epochs
        if cli_args.max_train_steps is not None:
            base_config.max_train_steps = cli_args.max_train_steps
        if cli_args.max_val_steps is not None:
            base_config.max_val_steps = cli_args.max_val_steps
        if cli_args.batch_size is not None:
            base_config.batch_size = cli_args.batch_size

    # Dynamically pull in the sweep parameters (like learning_rate) using our utility
    config = override_config_from_wandb(base_config, wandb.config)

    # Force the W&B dashboard to show the final, merged configuration
    wandb.config.update(config.to_wandb_dict(), allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = wandb.run.name if wandb.run else "local_run"
    print(f"Run: {run_name} | Device: {device} | Epochs: {config.epochs} | "
          f"LR: {config.learning_rate} | Batch: {config.batch_size}")

    # 3. Data & Model Initialization
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    dataset_path = cli_args.data_path if cli_args and cli_args.data_path else "data/processed/train_full.parquet"

    full_dataset = HMSequenceDataset(dataset_path, tokenizer, config)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True)

    model = HMGenRecModel(config, num_items=full_dataset.num_items).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    best_val_map = -1.0
    os.makedirs("models", exist_ok=True)

    # 4. Training Loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch in progress_bar:
            train_steps += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_idx = batch["target_idx"].to(device)

            outputs = model(input_ids, attention_mask, target_idx=target_idx)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            wandb.log({"train/batch_loss": loss.item()})

            if config.max_train_steps > 0 and train_steps >= config.max_train_steps:
                break

        epoch_train_loss = train_loss / max(1, train_steps)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        val_map_12 = 0.0
        val_steps = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]")
        with torch.no_grad():
            for batch in val_progress_bar:
                val_steps += 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_idx = batch["target_idx"].to(device)

                outputs = model(input_ids, attention_mask, target_idx=target_idx)
                loss = outputs["loss"]
                logits = outputs["logits"]

                batch_map = calculate_map_at_k(logits, target_idx, k=12)
                val_loss += loss.item()
                val_map_12 += batch_map
                val_progress_bar.set_postfix({"loss": loss.item(), "map@12": batch_map})

                if config.max_val_steps > 0 and val_steps >= config.max_val_steps:
                    break

        epoch_val_loss = val_loss / max(1, val_steps)
        epoch_val_map = val_map_12 / max(1, val_steps)

        # 6. Logging & Checkpoint Saving
        wandb.log({
            "train/epoch_loss": epoch_train_loss,
            "val/epoch_loss": epoch_val_loss,
            "val/map_at_12": epoch_val_map,
            "epoch": epoch
        })

        if epoch_val_map > best_val_map:
            print(f"\nNew best MAP@12: {best_val_map:.4f} -> {epoch_val_map:.4f}. Saving...")
            best_val_map = epoch_val_map

            state_dict_to_save = extract_trainable_state_dict(model)
            save_path = f"models/best_genrec_{run_name}.pt"
            torch.save(state_dict_to_save, save_path)

            # Optionally log to W&B artifacts since we are only saving the ~30MB adapter
            if wandb.run:
                artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/train_full.parquet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    # CRITICAL: parse_known_args() ignores the extra hyperparameter flags W&B injects,
    # preventing "unrecognized arguments" errors.
    args, unknown_args = parser.parse_known_args()

    # Print the unknown args so you can see exactly what W&B passed in the terminal!
    if unknown_args:
        print(f"Sweep injected hyperparameters: {unknown_args}")

    train(args)
