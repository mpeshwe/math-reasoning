import re
from src.data.load_gsm8k import load_gsm8k
from src.data.load_metamath import load_metamath
from src.data.load_rstar import load_rstar

def normalize_gsm8k(example, idx):
    """Convert a raw GSM8K example to clean schema."""
    raw = example["answer"]

    # Strip calculator annotations: <<48/2=24>> → ""
    clean = re.sub(r"<<.*?>>", "", raw)

    # Split on #### to separate solution from final answer
    parts = clean.split("####")
    solution = parts[0].strip()
    steps = [s.strip() for s in solution.split("\n") if s.strip()]
    final = parts[-1].strip()

    return {
        "id": f"gsm8k_{idx:05d}",
        "source": "gsm8k",
        "question": example["question"],
        "solution_text": solution,
        "solution_steps": steps,
        "final_answer": final,
        "num_steps": len(steps),
    }

def normalize_metamath(example, idx):
    """Convert a raw MetaMathQA example to clean schema."""
    response = example["response"]

    match = re.search(r"The answer is:\s*(.*)", response)
    if match:
        answer_str = match.group(1).strip().rstrip(".")
        solution = response[:match.start()].strip()
    else:
        answer_str = ""
        solution = response.strip()

    steps = [s.strip() for s in solution.split("\n") if s.strip()]

    return {
        "id": f"metamath_{idx:05d}",
        "source": "metamath",
        "question": example["query"],
        "solution_text": solution,
        "solution_steps": steps,
        "final_answer": answer_str,
        "num_steps": len(steps),
    }


def normalize_rstar(example, idx):
    """Convert a raw rStar SFT example to clean schema."""
    response = example["response"]

    # Extract final answer from \boxed{...} (handles nested braces)
    answer_str = ""
    match = re.search(r"\\boxed\{", response)
    if match:
        start = match.end()
        depth = 1
        i = start
        while i < len(response) and depth > 0:
            if response[i] == "{":
                depth += 1
            elif response[i] == "}":
                depth -= 1
            i += 1
        answer_str = response[start:i - 1].strip()

    # Extract solution (code block content)
    code_match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    solution = code_match.group(1).strip() if code_match else response.strip()

    # Split on <end_of_step> markers
    steps = [s.strip() for s in solution.split("<end_of_step>") if s.strip()]

    return {
        "id": f"rstar_{idx:06d}",
        "source": "rstar",
        "question": example["query"],
        "solution_text": solution,
        "solution_steps": steps,
        "final_answer": answer_str,
        "num_steps": len(steps),
    }


def build_clean_dataset(source="gsm8k", split="train", n_samples=-1):
    """Load raw dataset, normalize it, return clean Dataset."""
    if source == "gsm8k":
        raw = load_gsm8k(split)
        if n_samples > 0:
            raw = raw.select(range(min(n_samples, len(raw))))
        clean = raw.map(normalize_gsm8k, with_indices=True, remove_columns=raw.column_names)
    elif source == "metamath":
        raw = load_metamath(n_samples)
        clean = raw.map(normalize_metamath, with_indices=True, remove_columns=raw.column_names)
    elif source == "rstar":
        raw = load_rstar(n_samples)
        clean = raw.map(normalize_rstar, with_indices=True, remove_columns=raw.column_names)
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"[src/data/normalize.py] build_clean_dataset: Clean {source}: {len(clean)} examples")
    return clean