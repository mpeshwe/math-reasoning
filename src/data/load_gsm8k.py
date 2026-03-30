from datasets import load_dataset

def load_gsm8k(split="train") :
    """
    Loads the GSM8K dataset from HuggingFace.
    
    Args: 
        split(str) : "train" (7473 examples),  
                      "test" (1319 examples)
    """

    dataset = load_dataset("openai/gsm8k", "main" , split= split)
    print(f"[src/data/load_gsm8k.py] GSM8K {split}: {len(dataset)} examples")

    return dataset