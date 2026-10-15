import argparse
import json
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
    config = GenRecConfig()
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(config.backbone_name)
    data_dir = Path(args.data_dir)
    article_map, customer_map = load_metadata(data_dir)

    vocab_path = Path(args.vocab_path) if args.vocab_path else Path(args.model_path).with_name(
        Path(args.model_path).stem + "_vocab.json"
    )
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Training vocabulary not found: {vocab_path}. Use the JSON saved with the checkpoint "
            "so item embedding indices cannot silently drift."
        )
    with vocab_path.open(encoding="utf-8") as handle:
        idx2target = {int(idx): item for idx, item in json.load(handle).items()}
    num_items = len(idx2target)

    model = HMGenRecModel(config, num_items=num_items)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    transactions = pd.read_csv(
        data_dir / "raw" / "transactions_train.csv", dtype={"article_id": str}, parse_dates=["t_dat"]
    )
    history_dict = (
        transactions.sort_values(["customer_id", "t_dat", "article_id"])
        .groupby("customer_id")["article_id"].apply(list).to_dict()
    )
    del transactions
    submission_df = pd.read_csv(data_dir / "raw" / "sample_submission.csv")
    customers = submission_df["customer_id"].tolist()
    if args.max_users is not None:
        customers = customers[:args.max_users]

    dataset = HMInferenceDataset(customers_list=customers, history_dict=history_dict,
                                 article_map=article_map, customer_map=customer_map,
                                 tokenizer=tokenizer, max_length=config.max_seq_length)
    loader = DataLoader(dataset, batch_size=config.batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device), target_idx=None)
            indices = torch.topk(outputs["logits"], k=min(12, num_items), dim=1).indices.cpu().tolist()
            for customer_id, item_indices in zip(batch["customer_id"], indices):
                predictions.append({
                    "customer_id": customer_id,
                    "prediction": " ".join(str(idx2target[i]).zfill(10) for i in item_indices),
                })
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_path, index=False)
    print(f"Successfully wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Exact vocabulary JSON saved alongside the checkpoint")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="submissions/submission.csv")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignored unknown args: {unknown_args}")
    generate_submission(args)
