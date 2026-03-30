import re
import math


def extract_answer(text):
    """Extract the answer from model output text.

    Tries in order:
        1. #### delimiter
        2. "answer is" pattern
        3. Last number in text
        4. Return cleaned text (for symbolic answers)

    Returns:
        str — the extracted answer as a string, or "" if nothing found
    """
    if not text or not text.strip():
        return ""

    # Strategy 0: \boxed{} delimiter (rStar format)
    # Handle nested braces: \boxed{\frac{3}{4}} needs to match full content
    match = re.search(r"\\boxed\{", text)
    if match:
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        return text[start:i - 1].strip()

    # Strategy 1: #### delimiter
    if "####" in text:
        return text.split("####")[-1].strip()

    # Strategy 2: "answer is" pattern
    match = re.search(r"[Tt]he answer is[:\s]*(.*)", text)
    if match:
        ans = match.group(1).strip().rstrip(".")
        return ans

    # Strategy 3: last number in the text (including fractions like 2/3)
    fractions = re.findall(r"-?\d[\d,]*\.?\d*/\d[\d,]*\.?\d*", text)
    if fractions:
        return fractions[-1]
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1]

    # Strategy 4: return the last line, stripped
    return text.strip().split("\n")[-1].strip()


def parse_numeric(s):
    """Try to parse a string as a number. Returns float or None."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")

    # Handle fractions: "2/3"
    if "/" in s and not s.startswith("\\"):
        try:
            parts = s.split("/")
            return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            return None

    try:
        return float(s)
    except ValueError:
        return None


def normalize_symbolic(s):
    """Normalize a symbolic math expression for comparison.

    Handles: (2,-3), x=3 y=5, \frac{3}{4}+\sqrt{2}, 3 and 7, etc.
    """
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace("and", ",")
    s = s.replace("$", "")
    s = s.lower()

    # Sort comma-separated parts so (2,-3) matches (-3,2) order doesn't matter
    # But keep original if no commas
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) > 1:
        return ",".join(sorted(parts))

    return s


def is_correct(model_output, ground_truth):
    """Check if model output matches ground truth.

    Args:
        model_output: raw text from the model
        ground_truth: str — the correct answer

    Returns:
        bool
    """
    extracted = extract_answer(model_output)
    if not extracted and not ground_truth:
        return True
    if not extracted or not ground_truth:
        return False

    # Try numeric comparison first
    ext_num = parse_numeric(extracted)
    gt_num = parse_numeric(ground_truth)

    if ext_num is not None and gt_num is not None:
        return abs(ext_num - gt_num) < 0.01

    # Fall back to symbolic comparison
    return normalize_symbolic(extracted) == normalize_symbolic(ground_truth)