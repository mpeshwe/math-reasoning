# stdlib
import argparse
import json
import os
import sys

# Project root set (must be before src imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# third-party
import requests
import yaml

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


def generate_solutions(dataset, config):
    """Generate model outputs via vLLM server and evaluate correctness."""
    results = []
    prompt_template = config["prompt_template"]
    server_url = config["inference"]["server_url"]
    model_name = config["model"]["student"]["name"]

    for i, example in enumerate(dataset):
        prompt = prompt_template.format(question=example["question"])

        response = requests.post(f"{server_url}/v1/completions", json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": config["generation"]["max_new_tokens"],
            "temperature": config["generation"]["temperature"],
        })
        response.raise_for_status()
        model_output = response.json()["choices"][0]["text"]

        # Evaluate using Phase 1 code
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

        print(f"[run_baseline] generate_solutions: [{i+1}/{len(dataset)}] correct={correct} extracted={extracted} truth={ground_truth}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_test_dataset(config)

    print(f"[run_baseline] main: Loaded {len(dataset)} problems")

    results = generate_solutions(dataset, config)

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
