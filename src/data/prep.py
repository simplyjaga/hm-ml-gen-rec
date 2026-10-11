# src/data/prep.py
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def get_first_sentence(text: str) -> str:
    """Extracts the first sentence from a description."""
    text = str(text)
    if pd.isna(text) or text == 'nan' or not text.strip():
        return ""
    return text.split('.')[0].strip() + "." if '.' in text else text.strip()

def load_metadata(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Loads and formats rich article and customer metadata into dictionaries."""
    print("Loading metadata...")
    articles = pd.read_csv(data_dir / 'raw' / 'articles.csv', dtype={'article_id': str})
    customers = pd.read_csv(data_dir / 'raw' / 'customers.csv')

    articles['detail_desc'] = articles['detail_desc'].apply(get_first_sentence)

    # Fill NAs cleanly to prevent string concatenation errors
    cols_to_fill = [
        'prod_name', 'product_type_name', 'graphical_appearance_name',
        'colour_group_name', 'department_name', 'index_name'
    ]
    for col in cols_to_fill:
        articles[col] = articles[col].fillna('Unknown')

    # Construct a highly descriptive, natural language string
    articles['item_text'] = (
        articles['prod_name'] + " (" +
        articles['product_type_name'] + ", " +
        articles['graphical_appearance_name'] + " pattern, " +
        articles['colour_group_name'] + ", " +
        articles['department_name'] + " dept, " +
        articles['index_name'] + ") - " +
        articles['detail_desc']
    )
    # Using a fast dictionary lookup (O(1)) for mapping IDs later
    article_map = dict(zip(articles['article_id'], articles['item_text']))

    # Enhance Customer Profile
    customers['age'] = customers['age'].fillna(0).astype(int)
    customers['club_member_status'] = customers['club_member_status'].fillna('UNKNOWN')
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')

    customers['profile_text'] = (
        "Age: " + customers['age'].astype(str) +
        ", Status: " + customers['club_member_status'] +
        ", News Frequency: " + customers['fashion_news_frequency']
    )
    customer_map = dict(zip(customers['customer_id'], customers['profile_text']))

    return article_map, customer_map


def _process_user_history(args: tuple) -> list[dict]:
    """Helper function to run on separate CPU cores."""
    customer_id, item_list, max_context = args
    sequences = []

    if len(item_list) >= 2:
        for i in range(1, len(item_list)):
            target = item_list[i]
            start_idx = max(0, i - max_context)
            context = item_list[start_idx:i]

            sequences.append({
                'customer_id': customer_id,
                'context_ids': context,
                'target_id': target
            })
    return sequences

def build_sliding_window(transactions: pd.DataFrame, max_context: int = 5) -> pd.DataFrame:
    """Generates context->target pairs using a multi-core sliding window."""
    print("Aggregating user histories...")
    # Fast pandas vectorization to group items into lists
    grouped = (
        transactions.sort_values(['customer_id', 't_dat'])
        .groupby('customer_id')['article_id']
        .apply(list)
        .reset_index()
    )

    print(f"Distributing {len(grouped)} users across CPU cores...")

    # Using zip() is highly type-safe and avoids itertuples() attribute errors
    tasks = [
        (str(c_id), list(a_list), max_context)
        for c_id, a_list in zip(grouped['customer_id'], grouped['article_id'])
    ]

    # Use all available cores (leave 1 free to keep your system responsive)
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    sequences = []
    # Spin up worker processes
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map() returns results in the same order as tasks
        results = executor.map(_process_user_history, tasks, chunksize=1000)

        # Flatten the list of lists returned by the workers
        for user_sequences in results:
            sequences.extend(user_sequences)

    print(f"Generated {len(sequences)} sliding window sequences.")
    return pd.DataFrame(sequences)

def main(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    data_dir.joinpath('processed').mkdir(exist_ok=True)

    article_map, customer_map = load_metadata(data_dir)

    print("Loading and filtering transactions...")
    transactions_df = pd.read_csv(data_dir / 'raw' / 'transactions_train.csv', dtype={'article_id': str})
    assert isinstance(transactions_df, pd.DataFrame) # Tells the type checker: "Trust me, this is a DataFrame"

    # Filter to last ~4 weeks of data (dataset ends 2020-09-22)
    transactions_df = transactions_df[transactions_df['t_dat'] >= '2020-08-20'].copy()
    assert isinstance(transactions_df, pd.DataFrame)

    if args.sample_size > 0:
        print(f"Sampling {args.sample_size} unique users for local dev...")
        customer_series = transactions_df['customer_id']
        assert isinstance(customer_series, pd.Series)

        # .drop_duplicates() stays as a Series, which the type checker understands better than .unique()
        sample_users = customer_series.drop_duplicates().sample(n=args.sample_size, random_state=42)

        transactions_df = transactions_df[transactions_df['customer_id'].isin(sample_users)].copy()
        assert isinstance(transactions_df, pd.DataFrame)

        output_file = data_dir / 'processed' / 'train_sample.parquet'
    else:
        output_file = data_dir / 'processed' / 'train_full.parquet'

    seq_df = build_sliding_window(transactions_df, max_context=args.max_context)

    # Map IDs to actual text before saving using safe .apply()
    print("Mapping IDs to text representations...")
    seq_df['user_profile'] = seq_df['customer_id'].apply(lambda x: customer_map.get(x, "Unknown User"))

    # Pre-join the text with the newline and arrow directly here!
    seq_df['context_texts'] = seq_df['context_ids'].apply(
        lambda ids: "\n-> ".join([article_map.get(i, "Unknown Item") for i in ids])
    )
    # Convert context_ids list to a comma-separated string so PyArrow is happy
    seq_df['context_ids'] = seq_df['context_ids'].apply(lambda ids: ",".join(ids))
    seq_df['target_text'] = seq_df['target_id'].apply(lambda x: article_map.get(x, "Unknown Item"))

    seq_df.to_parquet(output_file, index=False)
    print(f"Saved {len(seq_df)} sequences to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--sample_size", type=int, default=500, help="Number of users for dev sample (0 for full dataset)")
    parser.add_argument("--max_context", type=int, default=5, help="Max items in history sequence")
    args = parser.parse_args()
    main(args)
