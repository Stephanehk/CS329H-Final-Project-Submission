import argparse
import re
import sys
from typing import List
import ast
import pickle

from utils.pandemic_gt_rew_fns import TruePandemicRewardFunction
from utils.traffic_gt_rew_fns import TrueTrafficRewardFunction

ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")  # strip ANSI color/format codes
PATTERN = re.compile(r"\*\*gt_rew_i:\s*(\d+)")  # captures digits after "**gt_rew_i:"


def extract_max_reward_and_path(logfile_path: str):
    MEAN_RET_RE        = re.compile(r"default_policy/modified_reward_mean'\s*:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)") #re.compile(r"Mean return over 5 episodes:\s*([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)")
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



env_name = "pandemic"

if env_name == "pandemic":
    gt_rew_fns = TruePandemicRewardFunction().get_all_weights()
    SLURM_IDS = [
        12769268, 12769269, 12737874, 12769270, 12769275, 12769279, 12769282, 12737880, 12737881, 12737882,
        12737884, 12737885, 12737886, 12737887, 12737888, 12737889, 12737890, 12737891, 12737892, 12737893,
        12737894, 12769283, 12769284, 12769287, 12769288, 12737899, 12737900, 12737901, 12737902, 12737904,
        12737905, 12737906, 12737907, 12737908, 12737909, 12737910, 12737911, 12737912, 12737913, 12737914,
        12737915, 12737918, 12737919, 12737920, 12737921, 12737922, 12737923, 12737924, 12737925, 12737928
    ]

    #these are rollouts from policies trained in larger towns so we can see infections go above threshold
    SLURM_IDS[46] = 12877528
    SLURM_IDS[47] = 12877529
    SLURM_IDS[48] = 12877530
    SLURM_IDS[49] = 12877531
    
    SLURM_IDS_flipped = SLURM_IDS[25:]
    SLURM_IDS = SLURM_IDS[:25]

elif env_name == "traffic":
    gt_rew_fns = TrueTrafficRewardFunction().get_all_weights()

    SLURM_IDS  = [
        12738089,12738091,12738092,12738093,12738094,12738095,12738096,12738097,12738098,12738099,12738100,12738101,12738102,12738103,12738104,12738105,12738106,12738107,12738108,12738109,12738110,12738111,12738112,12738113,12738114,12738115,12738116,12738117,12738118,12738119,12738120,12738121,12738124,12738125,12738126,12738127,12738128,12738129,12738130,12738131,12738132,12738133,12738134,12738135,12738136,12738137,12738138,12738139,12738140,12738141
    ]
    SLURM_IDS.append(12771536)
    
    #flipped sign of above:
    SLURM_IDS_flipped = SLURM_IDS[25:]
    SLURM_IDS = SLURM_IDS[:25]
    # print (SLURM_IDS_flipped)
    # assert False
gt_rew_fns += [ [-w for w in weights] for weights in gt_rew_fns ]  # add flipped sign versions
#needed for how we index keys below
assert len(SLURM_IDS) == 25
assert (len(SLURM_IDS_flipped) == 25 and env_name != "traffic") or (len(SLURM_IDS_flipped) == 26 and env_name == "traffic")

SLURM_IDS.extend(SLURM_IDS_flipped)


checkpoint_paths = {}
reward_fn_index2slurm_id = {}
reward_fn_index2max_reward = {}
for slurm_id in SLURM_IDS:
    #gt_rew_fn_run_logs
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
   
    key = matches[0]
    if slurm_id in SLURM_IDS_flipped and key < 25:
        key += 25

    print(f"Max Reward Mean: {max_reward} at Iteration {max_iter}")
    print(f"Checkpoint Path: {max_path}")
    print ("key:", key)
    print ("-"*50)


    checkpoint_paths[key] = max_path
    reward_fn_index2slurm_id[key] = slurm_id
    reward_fn_index2max_reward[key] = max_reward
#save each dict to file with pickle
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2checkpoint_paths.pkl", "wb") as f:
    pickle.dump(checkpoint_paths, f)
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2slurm_ids.pkl", "wb") as f:
    pickle.dump(reward_fn_index2slurm_id, f)
with open(f"data/gt_rew_fn_data/{env_name}_gt_rew_fns2max_reward.pkl", "wb") as f:
    pickle.dump(reward_fn_index2max_reward, f)


