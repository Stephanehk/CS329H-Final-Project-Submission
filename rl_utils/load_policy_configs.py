

import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.rllib.algorithms.ppo import PPO

checkpoint_path = "logs/data/mujoco/20251106_165904/checkpoint_9999"
state = Algorithm._checkpoint_info_to_algorithm_state(get_checkpoint_info(checkpoint_path))
# Extract the config and override GPU settings
config = state["config"].copy()

print (config)
#print the keys of the config
for k in config.keys():
    print (k)
    print (config[k])
    print ("-"*100)