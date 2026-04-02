# stdlib
import argparse
import json
import os
import sys

# Project root set (must be before src imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# third-party
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Phase 1 code
from src.data.normalize import build_clean_dataset
from src.eval.extract_answer import extract_answer, is_correct
from src.eval.metrics import compute_accuracy, accuracy_by_steps, save_results


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_test_dataset(config):
    """Load and normalize the test dataset."""
    n = config["data"]["eval_samples"] if config["data"]["eval_samples"] > 0 else -1
    test = build_clean_dataset("gsm8k", "test", n_samples=n)
    return test


def load_model_and_tokenizer(config):
    """Load model and tokenizer from config."""
    model_name = config["model"]["student"]["name"]
    device = config["model"]["student"]["device"]
    precision = config["model"]["student"]["precision"]

    if device == "cuda" and not torch.cuda.is_available():
        print("[run_baseline] load_model_and_tokenizer: CUDA not available, falling back to CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if precision == "fp16":
        model.half()
    elif precision == "bf16":
        model.bfloat16()

    model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_solutions(model, tokenizer, dataset, config, device):
    """Generate model outputs in batches and evaluate correctness."""
    results = []
    prompt_template = config["prompt_template"]
    batch_size = config.get("inference", {}).get("batch_size", 8)

    # Build all prompts
    prompts = [prompt_template.format(question=ex["question"]) for ex in dataset]

    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        batch_examples = [dataset[i] for i in range(start, end)]

        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config["generation"]["max_new_tokens"],
                do_sample=config["generation"]["do_sample"],
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each example in batch
        for j, example in enumerate(batch_examples):
            new_tokens = outputs[j, input_length:]
            model_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

            extracted = extract_answer(model_output)
            ground_truth = example["final_answer"]
            correct = is_correct(extracted, ground_truth)

            results.append({
                "id": example["id"],
                "question": example["question"],
                "ground_truth": ground_truth,
                "num_steps": example["num_steps"],
                "model_output": model_output,
                "extracted_answer": extracted,
                "is_correct": correct,
            })

            idx = start + j + 1
            print(f"[run_baseline] generate_solutions: [{idx}/{len(dataset)}] correct={correct} extracted={extracted} truth={ground_truth}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_test_dataset(config)
    model, tokenizer, device = load_model_and_tokenizer(config)

    print(f"[run_baseline] main: Loaded {len(dataset)} problems, model on {device}")

    results = generate_solutions(model, tokenizer, dataset, config, device)

    # Compute metrics
    accuracy = compute_accuracy(results)
    by_steps = accuracy_by_steps(results)
    metrics = {"accuracy": accuracy, "accuracy_by_steps": by_steps}

    print(f"[run_baseline] main: Accuracy = {accuracy:.2%} ({sum(r['is_correct'] for r in results)}/{len(results)})")
    print(f"[run_baseline] main: By steps = {by_steps}")

    # Save outputs (one JSON per line)
    os.makedirs("results", exist_ok=True)
    with open("results/baseline_outputs.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[run_baseline] main: Saved {len(results)} outputs to results/baseline_outputs.jsonl")

    # Save metrics
    save_results(results, metrics, "results/baseline_metrics.json")


if __name__ == "__main__":
    main()
