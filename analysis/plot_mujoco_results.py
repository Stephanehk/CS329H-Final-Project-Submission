#!/usr/bin/env python3
import argparse
import re
from typing import List
import matplotlib.pyplot as plt

def extract_episode_reward_means(text: str) -> List[float]:
    # Match numbers like 123, -1.23, 1.23e-4, etc., after episode_reward_mean (single or double quotes)
    num_pat = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    key_pat = rf"""["']episode_reward_mean["']\s*:\s*({num_pat})"""
    raw = [float(m) for m in re.findall(key_pat, text)]

    # RLlib usually prints the same mean twice per iteration (inside sampler_results and at top level).
    # Collapse consecutive duplicates so each iteration contributes one value.
    deduped = []
    last = object()
    for v in raw:
        if v != last:
            deduped.append(v)
            last = v
    return deduped


path = "run_logs/slurmjob_12970870.err"
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

vals = extract_episode_reward_means(text)
print(vals)
plt.plot(vals)
plt.savefig("mujoco_results.png")
plt.close()
