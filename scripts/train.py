import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import extract_trainable_state_dict, override_config_from_wandb

import wandb
from src.configs.model_config import GenRecConfig
from src.data.dataset import HMSequenceDataset
from src.model.genrec import HMGenRecModel
from src.training.metrics import calculate_map_at_k


def train(cli_args=None):
    wandb.init(project="hm_genrec", mode="online")
    base_config = GenRecConfig()
    if cli_args is not None:
        for field in ("epochs", "max_train_steps", "max_val_steps", "batch_size"):
            value = getattr(cli_args, field, None)
            if value is not None:
                setattr(base_config, field, value)
    config = override_config_from_wandb(base_config, wandb.config)
    wandb.config.update(config.to_wandb_dict(), allow_val_change=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = wandb.run.name if wandb.run else "local_run"
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    dataset_path = cli_args.data_path if cli_args and cli_args.data_path else "data/processed/train_full.parquet"

    rows = pd.read_parquet(dataset_path)
    if "target_date" not in rows.columns:
        raise ValueError("Prepared data has no target_date; rerun preprocessing before training.")
    rows["target_date"] = pd.to_datetime(rows["target_date"])
    last_date = rows["target_date"].max()
    validation_start = last_date - pd.Timedelta(days=6)
    train_rows = rows[rows["target_date"] < validation_start].copy()
    val_rows = rows[rows["target_date"] >= validation_start].copy()
    if train_rows.empty or val_rows.empty:
        raise ValueError("Chronological train/validation split is empty; check the prepared date range.")

    required_context_cols = {"val_context_ids", "val_context_texts"}
    if not required_context_cols.issubset(rows.columns):
        raise ValueError(
            "Prepared data lacks frozen validation contexts; rerun src.data.prep "
            "from the updated branch."
        )
    val_rows["context_ids"] = val_rows["val_context_ids"]
    val_rows["context_texts"] = val_rows["val_context_texts"]

    target2idx = {str(item): idx for idx, item in enumerate(train_rows["target_id"].astype(str).unique())}
    train_rows = train_rows[train_rows["target_id"].astype(str).isin(target2idx)].reset_index(drop=True)
    val_rows = val_rows[val_rows["target_id"].astype(str).isin(target2idx)].reset_index(drop=True)
    if train_rows.empty or val_rows.empty:
        raise ValueError("No validation targets appear in the training vocabulary.")
    print(f"Temporal split at {validation_start.date()}: {len(train_rows)} train rows, "
          f"{len(val_rows)} validation rows, {len(target2idx)} train-only items")

    train_data = HMSequenceDataset(dataset_path, tokenizer, config, target2idx, train_rows)
    val_data = HMSequenceDataset(dataset_path, tokenizer, config, target2idx, val_rows)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, pin_memory=True)
    model = HMGenRecModel(config, num_items=len(target2idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_score = -1.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        train_loss, train_steps = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for batch in progress:
            train_steps += 1
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            target_idx=batch["target_idx"].to(device))
            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()
            train_loss += outputs["loss"].item()
            progress.set_postfix(loss=outputs["loss"].item())
            wandb.log({"train/batch_loss": outputs["loss"].item()})
            if config.max_train_steps > 0 and train_steps >= config.max_train_steps:
                break

        model.eval()
        val_loss_sum, score_sum, n_examples, val_steps = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                val_steps += 1
                targets = batch["target_idx"].to(device)
                outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), target_idx=targets)
                count = targets.shape[0]
                val_loss_sum += outputs["loss"].item() * count
                score_sum += calculate_map_at_k(outputs["logits"], targets, k=12) * count
                n_examples += count
                if config.max_val_steps > 0 and val_steps >= config.max_val_steps:
                    break
        train_loss_epoch = train_loss / max(1, train_steps)
        val_loss_epoch = val_loss_sum / max(1, n_examples)
        # This is row-wise reciprocal-rank@12, not official customer-level H&M MAP@12.
        val_rr = score_sum / max(1, n_examples)
        wandb.log({"train/epoch_loss": train_loss_epoch, "val/epoch_loss": val_loss_epoch,
                   "val/row_reciprocal_rank_at_12": val_rr, "epoch": epoch})
        if val_rr > best_score:
            print(f"New best row reciprocal-rank@12: {best_score:.4f} -> {val_rr:.4f}; saving checkpoint")
            best_score = val_rr
            checkpoint_path = f"models/best_genrec_{run_name}.pt"
            vocab_path = f"models/best_genrec_{run_name}_vocab.json"
            torch.save(extract_trainable_state_dict(model), checkpoint_path)
            with open(vocab_path, "w", encoding="utf-8") as handle:
                json.dump({str(idx): item for item, idx in target2idx.items()}, handle)
            if wandb.run:
                artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="model")
                artifact.add_file(checkpoint_path)
                artifact.add_file(vocab_path)
                wandb.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed/train_full.parquet")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Sweep injected hyperparameters: {unknown_args}")
    train(args)
