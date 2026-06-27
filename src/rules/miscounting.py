def check_miscounting_hallucination(claimed_count, true_count, start_idx, end_idx):
    if claimed_count != true_count:
        return {
            "is_hallucination": True,
            "category": "miscounting",
            "span": [start_idx, end_idx],
            "reason": f"Model claimed there are {claimed_count} items, but there are actually {true_count}.",
        }

    return {"is_hallucination": False}