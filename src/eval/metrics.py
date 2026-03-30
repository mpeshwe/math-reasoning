import json
from collections import defaultdict


def compute_accuracy(results):
    """Overall accuracy.

    Args:
        results: list of dicts, each with "is_correct" (bool)
    """
    correct = sum(1 for r in results if r["is_correct"])
    return correct / len(results) if results else 0.0


def accuracy_by_steps(results):
    """Accuracy grouped by problem difficulty (num_steps).

    Args:
        results: list of dicts, each with "is_correct" and "num_steps"

    Returns:
        dict[int, float] — {num_steps: accuracy}
    """
    groups = defaultdict(list)
    for r in results:
        groups[r["num_steps"]].append(r["is_correct"])

    return {
        steps: sum(correct) / len(correct)
        for steps, correct in sorted(groups.items())
    }


def save_results(results, metrics, path):
    """Save metrics to JSON for cross-phase comparison.

    Args:
        results: list of result dicts
        metrics: dict with "accuracy" and "accuracy_by_steps"
        path: output file path (e.g. "results/baseline_metrics.json")
    """
    output = {
        "accuracy": metrics["accuracy"],
        "accuracy_by_steps": metrics["accuracy_by_steps"],
        "total": len(results),
        "correct": sum(1 for r in results if r["is_correct"]),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[src/eval/metrics.py] save_results: Saved to {path}")
