from datasets import load_dataset


def load_rstar(n_samples=-1):
    """Load rStar SFT dataset from HuggingFace.

    Args:
        n_samples: Number of samples to load. -1 for all (~1.19M).
                   Use 50-100 for local testing.

    Returns:
        HuggingFace Dataset with 'query' and 'response' fields
    """
    ds = load_dataset("ElonTusk2001/rstar_sft", split="train")

    if n_samples > 0:
        ds = ds.select(range(min(n_samples, len(ds))))

    print(f"[src/data/load_rstar.py] load_rstar: {len(ds)} examples loaded")
    return ds
