#!/usr/bin/env python3
import re
import sys
from pathlib import Path
import numpy as np

def extract_metrics(text: str):
    # Matches lines like:
    # "Accuracy over training set, indifferent prefs. removed:: 0.999"
    acc_pattern = re.compile(
        r"Accuracy over training set.*?([0-9]+\.[0-9]+)"
    )

    # Matches lines like:
    # "Global Kendall-Tau: 0.908  (p=0.00e+00)"
    # Be robust to the various dash characters between Kendall and Tau
    kendall_pattern = re.compile(
        r"Global\s+Kendall[^\d\-+]*([0-9]+\.[0-9]+)"
    )

    accuracies = [float(m.group(1)) for m in acc_pattern.finditer(text)]
    kendalls = [float(m.group(1)) for m in kendall_pattern.finditer(text)]

    return accuracies, kendalls


# path = "run_logs/slurmjob_12877541.out"
# path = "pandemic_raw_ins_baseline_out.txt"
path = "traffic_raw_ins_baseline_out.txt"

path = Path(path)
if not path.is_file():
    print(f"File not found: {path}", file=sys.stderr)
    sys.exit(1)

text = path.read_text(encoding="utf-8", errors="ignore")

accuracies, kendalls = extract_metrics(text)

# print("Accuracies:")
# for a in accuracies:
#     print(a)

# print("\nKendall-Taus:")
# for k in kendalls:
#     print(k)

# If you want them as Python lists:
print("accuracies =", accuracies)
print("kendalls   =", kendalls)

print("mean accuracy =", np.mean(accuracies))
print("mean kendall =", np.mean(kendalls))
print ("--------------------------------")
print("min accuracy =", min(accuracies))
print("min kendall =", min(kendalls))
print ("--------------------------------")
print("max accuracy =", max(accuracies))
print("max kendall =", max(kendalls))
print ("--------------------------------")
print("std accuracy =", np.std(accuracies))
print("std kendall =", np.std(kendalls))
print ("--------------------------------")
print("median accuracy =", np.median(accuracies))
print("median kendall =", np.median(kendalls))
