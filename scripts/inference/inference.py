import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.inference.inference_dataset import HMInferenceDataset
from src.configs.model_config import GenRecConfig
from src.data.prep import load_metadata
from src.model.genrec import HMGenRecModel


def generate_submission(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    # 1. Setup Config & Tokenizer
    config = GenRecConfig()
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)

    data_dir = Path(args.data_dir)
    article_map, customer_map = load_metadata(data_dir)

    # 2. Rebuild Target Vocab (Guarantees exact parity with training indices)
    print(f"Loading unique targets from {args.train_data_path} to reconstruct vocab...")
    train_df = pd.read_parquet(args.train_data_path)
    unique_targets: list[str] = train_df["target_id"].unique().tolist()
    idx2target: dict[int, str] = {idx: target_id for idx, target_id in enumerate(unique_targets)}
    num_items: int = len(unique_targets)
    del train_df

    # 3. Initialize Model and Load Trained Checkpoint
    print("Loading model and checkpoint weights...")
    model = HMGenRecModel(config, num_items=num_items)

    # strict=False because only trainable parameters (LoRA, heads, embeddings) were saved
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    # 4. Prepare Transaction Histories
    print("Extracting user purchase histories...")
    transactions_df = pd.read_csv(data_dir / "raw" / "transactions_train.csv", dtype={"article_id": str})
    history_dict: dict[str, list[str]] = (
        transactions_df.sort_values(["customer_id", "t_dat"])
        .groupby("customer_id")["article_id"]
        .apply(list)
        .to_dict()
    )
    del transactions_df

    # 5. Load Target Users (From Kaggle sample_submission)
    print("Loading target submission users...")
    submission_df = pd.read_csv(data_dir / "raw" / "sample_submission.csv")
    customers_to_predict: list[str] = submission_df["customer_id"].tolist()

    if args.max_users is not None:
        print(f"Truncating inference to first {args.max_users} users for verification...")
        customers_to_predict = customers_to_predict[: args.max_users]

    dataset = HMInferenceDataset(
        customers_list=customers_to_predict,
        history_dict=history_dict,
        article_map=article_map,
        customer_map=customer_map,
        tokenizer=tokenizer,
        max_length=config.max_seq_length,
    )

    # Use 2x batch size for inference since backward pass activations aren't stored
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 6. Prediction Loop
    print("Generating top-12 predictions...")
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            customer_ids = batch["customer_id"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask, target_idx=None)
            logits = outputs["logits"]

            # Retrieve top 12 items
            _, top_indices = torch.topk(logits, k=12, dim=1)
            top_indices = top_indices.cpu().numpy()

            for i, c_id in enumerate(customer_ids):
                # Ensure 10-digit padded strings for H&M article IDs
                pred_strings = [str(idx2target[idx]).zfill(10) for idx in top_indices[i]]
                predictions.append({
                    "customer_id": c_id,
                    "prediction": " ".join(pred_strings)
                })

    # 7. Write Output CSV
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print(f"Successfully written {len(predictions)} predictions to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/processed/train_sample.parquet",
        help="Path to training parquet to restore catalog indexing",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="submissions/submission.csv",
        help="Target output CSV file path",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Limit user count for quick local testing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size override for inference",
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignored unknown args: {unknown_args}")

    generate_submission(args)
