# src/data/prep.py
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def get_first_sentence(text: str) -> str:
    """Extract the first sentence from a description."""
    text = str(text)
    if pd.isna(text) or text == "nan" or not text.strip():
        return ""
    return text.split(".")[0].strip() + "." if "." in text else text.strip()


def load_metadata(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load article descriptions and customer profile text."""
    print("Loading metadata...")
    articles = pd.read_csv(data_dir / "raw" / "articles.csv", dtype={"article_id": str})
    customers = pd.read_csv(data_dir / "raw" / "customers.csv")

    articles["detail_desc"] = articles["detail_desc"].apply(get_first_sentence)
    cols_to_fill = [
        "prod_name", "product_type_name", "graphical_appearance_name",
        "colour_group_name", "department_name", "index_name",
    ]
    for col in cols_to_fill:
        articles[col] = articles[col].fillna("Unknown")

    articles["item_text"] = (
        articles["prod_name"] + " (" + articles["product_type_name"] + ", "
        + articles["graphical_appearance_name"] + " pattern, "
        + articles["colour_group_name"] + ", " + articles["department_name"]
        + " dept, " + articles["index_name"] + ") - " + articles["detail_desc"]
    )
    article_map = dict(zip(articles["article_id"], articles["item_text"]))

    customers["age"] = customers["age"].fillna(0).astype(int)
    customers["club_member_status"] = customers["club_member_status"].fillna("UNKNOWN")
    customers["fashion_news_frequency"] = customers["fashion_news_frequency"].fillna("NONE")
    customers["profile_text"] = (
        "Age: " + customers["age"].astype(str)
        + ", Status: " + customers["club_member_status"]
        + ", News Frequency: " + customers["fashion_news_frequency"]
    )
    customer_map = dict(zip(customers["customer_id"], customers["profile_text"]))
    return article_map, customer_map


def _process_user_history(args: tuple) -> list[dict]:
    """Build next-day targets using only purchases from strictly earlier dates."""
    customer_id, day_records, max_context = args
    sequences = []
    history = []

    # All purchases on one date share the same context; within-day order is
    # not treated as a meaningful sequence.
    for target_date, items_bought_that_day in day_records:
        context = history[-max_context:]
        if context:
            for target in items_bought_that_day:
                sequences.append({
                    "customer_id": customer_id,
                    "context_ids": list(context),
                    "target_id": target,
                    "target_date": str(target_date),
                })
        history.extend(items_bought_that_day)
    return sequences


def build_sliding_window(transactions: pd.DataFrame, max_context: int = 5) -> pd.DataFrame:
    """Create context -> target rows without ordering purchases within a day."""
    print("Aggregating customer histories...")
    ordered = transactions.sort_values(["customer_id", "t_dat", "article_id"])
    tasks = []
    for customer_id, customer_rows in ordered.groupby("customer_id", sort=False):
        day_records = [
            (str(date.date()), day_rows["article_id"].astype(str).tolist())
            for date, day_rows in customer_rows.groupby("t_dat", sort=True)
        ]
        tasks.append((str(customer_id), day_records, max_context))

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    sequences = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for user_sequences in executor.map(_process_user_history, tasks, chunksize=1000):
            sequences.extend(user_sequences)

    print(f"Generated {len(sequences)} context-target sequences.")
    return pd.DataFrame(
        sequences,
        columns=["customer_id", "context_ids", "target_id", "target_date"],
    )


def main(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    article_map, customer_map = load_metadata(data_dir)

    print("Loading transactions...")
    transactions = pd.read_csv(
        data_dir / "raw" / "transactions_train.csv",
        dtype={"article_id": str},
        parse_dates=["t_dat"],
    )
    # Recent history window for training examples; validation cutoff is chosen
    # by train.py from the latest date in the raw transaction file.
    transactions = transactions[
        transactions["t_dat"] >= pd.Timestamp(args.start_date)
    ].copy()

    if args.sample_size > 0:
        print(f"Sampling {args.sample_size} customers for local development...")
        sampled = transactions["customer_id"].drop_duplicates().sample(
            n=min(args.sample_size, transactions["customer_id"].nunique()),
            random_state=42,
        )
        transactions = transactions[transactions["customer_id"].isin(sampled)].copy()
        output_file = data_dir / "processed" / "train_sample.parquet"
    else:
        output_file = data_dir / "processed" / "train_full.parquet"

    sequences = build_sliding_window(transactions, max_context=args.max_context)
    if sequences.empty:
        raise ValueError("No training sequences were generated; check the input dates.")

    # Freeze validation contexts at the start of the final 7-day target window.
    # Validation examples must not consume purchases made earlier in that same window.
    validation_start = transactions["t_dat"].max().normalize() - pd.Timedelta(days=6)
    history_before_cutoff = transactions[transactions["t_dat"] < validation_start].sort_values(
        ["customer_id", "t_dat", "article_id"]
    )
    cutoff_contexts = (
        history_before_cutoff.groupby("customer_id")["article_id"]
        .apply(lambda values: values.astype(str).tolist()[-args.max_context:])
        .to_dict()
    )
    sequences["val_context_ids"] = sequences["customer_id"].map(
        lambda cid: cutoff_contexts.get(cid, [])
    )
    sequences["val_context_texts"] = sequences["val_context_ids"].apply(
        lambda ids: "\n-> ".join(article_map.get(item, "Unknown Item") for item in ids)
    )
    sequences["val_context_ids"] = sequences["val_context_ids"].apply(lambda ids: ",".join(ids))

    sequences["user_profile"] = sequences["customer_id"].map(customer_map).fillna("Unknown User")
    sequences["context_texts"] = sequences["context_ids"].apply(
        lambda ids: "\n-> ".join(article_map.get(item, "Unknown Item") for item in ids)
    )
    sequences["context_ids"] = sequences["context_ids"].apply(lambda ids: ",".join(ids))
    sequences["target_text"] = sequences["target_id"].map(article_map).fillna("Unknown Item")
    sequences.to_parquet(output_file, index=False)
    print(f"Saved {len(sequences)} sequences to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of customers; 0 uses all")
    parser.add_argument("--max_context", type=int, default=5)
    parser.add_argument("--start_date", type=str, default="2020-08-20")
    main(parser.parse_args())
