import torch


def calculate_map_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int = 12) -> float:
    """Single-target AP@K (reciprocal rank); retained for row-wise diagnostics."""
    k = min(k, logits.shape[1])
    top_indices = torch.topk(logits, k, dim=1).indices
    hits = (top_indices == targets.unsqueeze(1)).float()
    weights = 1.0 / torch.arange(1, k + 1, device=logits.device, dtype=torch.float)
    return (hits * weights).sum(dim=1).mean().item()


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int = 12) -> float:
    """Compute AP@K for one customer with multiple relevant articles."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def mean_average_precision_at_k(
    recommendations: dict[str, list[str]],
    relevant_items: dict[str, set[str]],
    k: int = 12,
) -> float:
    """Customer-level MAP@K, excluding customers with no relevant purchases."""
    scores = [
        average_precision_at_k(recommendations[cid], relevant)
        for cid, relevant in relevant_items.items()
        if relevant and cid in recommendations
    ]
    return sum(scores) / len(scores) if scores else 0.0
