from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class HMInferenceDataset(Dataset):
    """
    Dataset for generating submission predictions (exactly one sequence per user).
    Builds context from historical transactions and applies causal padding/truncation.
    """
    def __init__(
        self,
        customers_list: list[str],
        history_dict: dict[str, list[str]],
        article_map: dict[str, str],
        customer_map: dict[str, str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
        max_context: int = 5,
    ) -> None:
        self.customers: list[str] = customers_list
        self.history_dict: dict[str, list[str]] = history_dict
        self.article_map: dict[str, str] = article_map
        self.customer_map: dict[str, str] = customer_map
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_length: int = max_length
        self.max_context: int = max_context

        # Enforce left padding and truncation to retain the final prompt tokens
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.customers)

    def __getitem__(self, idx: int):
        customer_id: str = self.customers[idx]
        user_profile: str = self.customer_map.get(customer_id, "Unknown User")

        # Slice the most recent context purchases
        history_ids: list[str] = self.history_dict.get(customer_id, [])[-self.max_context:]
        context_texts: str = "\n-> ".join(
            [self.article_map.get(item_id, "Unknown Item") for item_id in history_ids]
        )

        prompt: str = (
            f"Customer Profile: {user_profile}\n"
            f"Purchase History:\n{context_texts}\n\n"
            f"Next likely purchase:"
        )

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "customer_id": customer_id,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
