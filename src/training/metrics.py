# src/training/metrics.py
import torch


def calculate_map_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 12) -> float:
    """
    Calculates Mean Average Precision at K (MAP@K) for a batch.
    Since our validation rows currently have a single target item,
    this is mathematically equivalent to Mean Reciprocal Rank at K (MRR@K).

    Args:
        logits: Tensor of shape (batch_size, num_items)
        targets: Tensor of shape (batch_size,) containing the true item index
        k: The cutoff rank (default 12 for H&M Kaggle)

    Returns:
        float: The MAP@K score for the batch.
    """
    # 1. Get the indices of the top-k predicted items
    # shape: (batch_size, k)
    _, top_indices = torch.topk(logits, k, dim=1)

    # 2. Check if the true target is in the top-k predictions
    # Expand targets to (batch_size, 1) to broadcast against (batch_size, k)
    # shape: (batch_size, k) - boolean tensor
    hits = (top_indices == targets.unsqueeze(1)).float()

    # 3. Calculate the reciprocal rank weights: [1/1, 1/2, 1/3, ..., 1/k]
    # shape: (k,)
    weights = 1.0 / torch.arange(1, k + 1, device=logits.device, dtype=torch.float)

    # 4. Multiply hits by weights
    # If the target is at rank 3, hits is [0, 0, 1, 0...], weight is 1/3. Score = 1/3.
    ap_scores = torch.sum(hits * weights, dim=1)

    # 5. Return the mean across the batch
    return ap_scores.mean().item()
