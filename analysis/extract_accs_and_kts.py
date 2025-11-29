#!/usr/bin/env python3
import argparse
import math
import re
from statistics import mean

def parse_file(path: str):
    """
    Returns:
      accuracies_raw: list of floats or None (for N/A)
      kts_raw:        list of floats or None (for nan)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Accuracy lines look like:
    # "Accuracy over training set, indifferent prefs. removed:: 0.995"
    # Sometimes "N/A"
    acc_re = re.compile(
        r"Accuracy over training set.*?::\s*(?P<val>(?:\d+(?:\.\d+)?|N/?A))",
        re.IGNORECASE | re.DOTALL
    )

    # Kendall–Tau lines look like:
    # "Global Kendall-Tau: 0.884  (p=0.00e+00)"
    # Could contain various hyphen characters; sometimes "nan"
    kt_re = re.compile(
        r"Global\s+Kendall[\-\u2010\u2011\u2012\u2013\u2212]?\s*Tau:\s*(?P<val>(?:[-+]?\d+(?:\.\d+)?|nan))",
        re.IGNORECASE
    )

    accuracies_raw = []
    for m in acc_re.finditer(text):
        v = m.group("val").strip()
        if v.lower().replace("/", "") == "na":
            accuracies_raw.append(None)
        else:
            try:
                accuracies_raw.append(float(v))
            except ValueError:
                accuracies_raw.append(None)

    kts_raw = []
    for m in kt_re.finditer(text):
        v = m.group("val").strip().lower()
        if v == "nan":
            kts_raw.append(None)
        else:
            try:
                kts_raw.append(float(v))
            except ValueError:
                kts_raw.append(None)

    return accuracies_raw, kts_raw

def summarize(name: str, values):
    """Compute count, valid count, mean, min; ignore None and NaN."""
    cleaned = [v for v in values if v is not None and not math.isnan(v)]
    print(f"\n{name}:")
    # print(f"  total extracted: {len(values)}")
    # print(f"  valid (numeric): {len(cleaned)}")
    if cleaned:
        print(f"  mean: {mean(cleaned):.6f}")
        print(f"  min:  {min(cleaned):.6f}")
    else:
        print("  mean: N/A")
        print("  min:  N/A")



accuracies_raw, kts_raw = parse_file("run_logs/slurmjob_12869965.out")

# Print the raw lists (optional). Comment out if you prefer quieter output.
print("Accuracies (raw, including None for N/A):")
print(accuracies_raw)
print("\nKendall–Tau correlations (raw, including None for nan):")
print(kts_raw)

summarize("Accuracy", accuracies_raw)
summarize("Kendall–Tau", kts_raw)