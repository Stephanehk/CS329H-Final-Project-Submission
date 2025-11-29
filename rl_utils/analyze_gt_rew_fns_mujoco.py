import argparse
import re
import sys
from typing import List
import ast
import pickle
import numpy as np

from utils.mujoco_gt_rew_fns import TrueMujocoAntRewardFunction

ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")  # strip ANSI color/format codes
PATTERN = re.compile(r"\*\*gt_rew_i:\s*(\d+)")  # captures digits after "**gt_rew_i:"


def extract_max_reward_and_path(logfile_path: str):
    MEAN_RET_RE        = re.compile(r"episode_reward_mean'\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)") #re.compile(r"Mean return over 5 episodes:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
    ITER_RE            = re.compile(r"Iteration\s+(\d+)\s*/")  # e.g. "Iteration 1/100"
    CKPT_RE = re.compile(r"path=([^,\)]+checkpoint_\d+)", re.IGNORECASE)

    results = {}  # iteration -> (reward_mean, checkpoint_path)

    with open(logfile_path, "r") as f:
        lines = f.readlines()

    current_iter = None
    current_reward = None
    checkpoint_path = None

    for line in lines:
        
        # Detect iteration
        iter_match = ITER_RE.match(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
            current_reward = None
            checkpoint_path = None
            # continue

        # Detect metrics dict
        if (m := MEAN_RET_RE.search(line)):
            current_reward = float(m.group(1))
            # continue
        # Detect checkpoint
        if "checkpoint to TrainingResult" in line:
            # print ("Checkpoint line:", line[:500])
            m_ckpt = CKPT_RE.search(line)
            # print(m_ckpt.group(1))
            if m_ckpt:
                checkpoint_path = m_ckpt.group(1)

        # Store result if iteration divisible by 10
        if current_iter is not None and current_iter % 10 == 0 and current_reward is not None:
            results[current_iter] = (current_reward, checkpoint_path)

    # Find max reward mean
    max_iter = max(results, key=lambda i: results[i][0])
    max_reward, max_path = results[max_iter]

    return max_iter, max_reward, max_path

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def strip_ansi(s: str) -> str:
    return ANSI_ESCAPE_RE.sub("", s)

def extract_all(text: str) -> List[int]:
    return [int(m) for m in PATTERN.findall(text)]



env_name = "mujoco"
SLURM_IDS = [12982379, 12982520, 12982521]
np.random.seed(42)

checkpoint_paths = {}
reward_fn_index2slurm_id = {}
for slurm_id in SLURM_IDS:
    log_file_path = "/next/u/stephhk/llm_reward_design/run_logs/slurmjob_" + str(slurm_id) + ".err"
    raw = read_text(log_file_path)
    text = strip_ansi(raw)
    matches = extract_all(text)
    assert len(matches) == 1, f"Expected exactly one match in slurm ID {slurm_id}, found {len(matches)}"

    if not matches:
        raise ValueError(f"No matches found in slurm ID {slurm_id}")
    
    print ("Slurm ID:", slurm_id)
    # print ("Reward Function:", gt_rew_fns[matches[0]])

    max_iter, max_reward, max_path = extract_max_reward_and_path(log_file_path)
    print(f"Max Reward Mean: {max_reward} at Iteration {max_iter}")
    print(f"Checkpoint Path: {max_path}")
    print ("-"*50)

    key = matches[0]
    checkpoint_paths[key] = max_path
    reward_fn_index2slurm_id[key] = slurm_id

    if key == 0 or key == 1:
        #sample 24 other checkpoints from the same slurm job
        other_checkpoint_iters = [i for i in range(0, max_iter + 1) if i % 10 == 0]
        other_checkpoint_iters = np.random.choice(other_checkpoint_iters, size=23 + key, replace=False)
        for other_iter in other_checkpoint_iters:
            other_checkpoint_path =max_path.replace(max_path.split("/")[-1], "") + max_path.split("/")[-1].replace(str(max_iter), str(other_iter))
            print (other_checkpoint_path)
            checkpoint_paths[f"{key}_checkpoint_{other_iter}"] = other_checkpoint_path
            # reward_fn_index2slurm_id[key + i] = slurm_id

    # print ("key:", key)

#save each dict to file with pickle
print ("Total checkpoint paths:", len(checkpoint_paths))
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2checkpoint_paths.pkl", "wb") as f:
    pickle.dump(checkpoint_paths, f)
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2slurm_ids.pkl", "wb") as f:
    pickle.dump(reward_fn_index2slurm_id, f)


