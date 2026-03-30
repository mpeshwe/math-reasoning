"""Phase 1 integration test — run this to verify everything works."""

from src.data.normalize import build_clean_dataset
from src.eval.extract_answer import extract_answer, is_correct, parse_numeric
from src.eval.metrics import compute_accuracy, accuracy_by_steps, save_results

PASS = 0
FAIL = 0


def check(name, result, expected):
    global PASS, FAIL
    if result == expected:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — got {repr(result)}, expected {repr(expected)}")


# ============================================================
# 1. extract_answer tests
# ============================================================
print("\n[test_pipeline.py] test_extract_answer:")

# \boxed{} format (rStar)
check("boxed integer", extract_answer("\\boxed{42}"), "42")
check("boxed symbolic", extract_answer("\\boxed{\\sqrt{5}}"), "\\sqrt{5}")
check("boxed fraction", extract_answer("\\boxed{\\frac{3}{4}}"), "\\frac{3}{4}")

# #### format (GSM8K)
check("hash 18", extract_answer("#### 18"), "18")
check("hash 72", extract_answer("some work\n#### 72"), "72")

# "answer is" format (MetaMath)
check("answer is 72", extract_answer("The answer is: 72"), "72")
check("answer is 5.0", extract_answer("The answer is 5.0"), "5.0")

# Last number fallback
check("dollar 18", extract_answer("She makes $18 per day"), "18")
check("decimal", extract_answer("18.00"), "18.00")
check("negative", extract_answer("The total is -5"), "-5")
check("comma number", extract_answer("1,000 students"), "1,000")

# Edge cases
check("empty string", extract_answer(""), "")
check("no number", extract_answer("I cannot solve this"), "I cannot solve this")

# ============================================================
# 2. parse_numeric tests
# ============================================================
print("\n[test_pipeline.py] test_parse_numeric:")

check("int string", parse_numeric("72"), 72.0)
check("float string", parse_numeric("5.0"), 5.0)
check("negative", parse_numeric("-5"), -5.0)
check("comma", parse_numeric("1,000"), 1000.0)
check("dollar", parse_numeric("$18"), 18.0)
check("fraction", parse_numeric("2/3"), 2 / 3)
check("symbolic none", parse_numeric("\\sqrt{5}"), None)
check("text none", parse_numeric("hello"), None)

# ============================================================
# 3. is_correct tests
# ============================================================
print("\n[test_pipeline.py] test_is_correct:")

# Numeric matches
check("exact int", is_correct("#### 18", "18"), True)
check("float match", is_correct("#### 18.0", "18"), True)
check("wrong answer", is_correct("#### 19", "18"), False)
check("answer is format", is_correct("The answer is 72", "72"), True)
check("comma vs plain", is_correct("$1,000 total", "1000"), True)
check("negative match", is_correct("result is -5", "-5"), True)
check("fraction match", is_correct("about 2/3", "0.6667"), True)

# Symbolic matches
check("sqrt match", is_correct("\\boxed{\\sqrt{5}}", "\\sqrt{5}"), True)
check("frac match", is_correct("\\boxed{\\frac{3}{4}}", "\\frac{3}{4}"), True)

# Multi-part answers
check("coordinate pair", is_correct("\\boxed{(2, -3)}", "(2, -3)"), True)
check("multi variable", is_correct("\\boxed{x=3, y=5}", "x=3, y=5"), True)
check("expression", is_correct("\\boxed{\\frac{3}{4} + \\sqrt{2}}", "\\frac{3}{4} + \\sqrt{2}"), True)
check("two numbers", is_correct("#### 3 and 7", "3 and 7"), True)

# Failures
check("empty vs number", is_correct("", "18"), False)

# ============================================================
# 4. metrics tests
# ============================================================
print("\n[test_pipeline.py] test_metrics:")

results = [
    {"is_correct": True, "num_steps": 2},
    {"is_correct": True, "num_steps": 2},
    {"is_correct": False, "num_steps": 2},
    {"is_correct": True, "num_steps": 3},
    {"is_correct": False, "num_steps": 3},
    {"is_correct": False, "num_steps": 4},
]

check("overall accuracy", round(compute_accuracy(results), 4), 0.5)
by_steps = accuracy_by_steps(results)
check("2-step accuracy", round(by_steps[2], 4), round(2 / 3, 4))
check("3-step accuracy", round(by_steps[3], 4), 0.5)
check("4-step accuracy", round(by_steps[4], 4), 0.0)

metrics = {"accuracy": compute_accuracy(results), "accuracy_by_steps": by_steps}
save_results(results, metrics, "results/test_metrics.json")
check("save_results runs", True, True)

# ============================================================
# 5. Dataset loading + normalization (small samples)
# ============================================================
print("\n[test_pipeline.py] test_datasets:")

SCHEMA_KEYS = {"id", "source", "question", "solution_text", "solution_steps", "final_answer", "num_steps"}

for source in ["gsm8k", "metamath", "rstar"]:
    ds = build_clean_dataset(source, n_samples=10)
    check(f"{source} loads", len(ds), 10)
    check(f"{source} schema", set(ds[0].keys()), SCHEMA_KEYS)
    check(f"{source} has answer", bool(ds[0]["final_answer"]), True)
    check(f"{source} has steps", ds[0]["num_steps"] > 0, True)

# ============================================================
# 6. Round-trip: clean data → extract_answer → is_correct
# ============================================================
print("\n[test_pipeline.py] test_round_trip:")

gsm = build_clean_dataset("gsm8k", "train", n_samples=50)
correct = 0
for i in range(len(gsm)):
    ans = gsm[i]["final_answer"]
    # Simulate model echoing the answer in #### format
    fake_output = f"some reasoning\n#### {ans}"
    if is_correct(fake_output, ans):
        correct += 1
check("gsm8k round-trip (fake perfect model)", correct, 50)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

if FAIL == 0:
    print("  Phase 1 pipeline is solid.")
else:
    print("  FIX THE FAILURES ABOVE.")
