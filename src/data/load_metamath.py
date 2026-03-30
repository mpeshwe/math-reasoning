from datasets import load_dataset

def load_metamath(n_samples= -1) :
    """
    Load MetaMathQA dataset from HuggingFace.

    Args: 
        n_samples: Number of samples to load. If -1, load the entire dataset.
    Returns:
        HuggingFace Dataset with 'query', 'response', and 'type' 'original_question' fields
    """

    dataset = load_dataset("meta-math/MetaMathQA", split="train")
    if n_samples > 0:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    print(f"[src/data/load_metamath.py] MetaMathQA: {len(dataset)} examples")
    return dataset