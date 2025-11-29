
# Flow Traffic Environment
This code was adapted for "Preventing Reward Hacking using Occupancy Measure Regularization". More details about how to run experiments can be found within our main repository and our paper. Support for this environment has stopped, so we have revived the code with some of the latest package versions. Additionally, we edited this environment so that the safe policy actions can be returned to the RL policies we train, and the true and proxy reward functions can be calculated simultaneously. 

This repository is based on the [code](https://github.com/aypan17/reward-misspecification/tree/main/flow) of [Pan et al.](https://arxiv.org/abs/2201.03544). 

The original code release for the Flow traffic environment can be found [here](https://github.com/flow-project/flow).

## Installation
Running 
```
pip install -r requirements.txt
```
from our main repository will install this package along with all of its depedencies.

Additioally, instructions for installing SUMO, which is required for Flow, can be found [here](https://flow.readthedocs.io/en/latest/flow_setup.html).

## Citing Flow

If you use Flow for academic research, you are highly encouraged to cite our paper:

C. Wu, A. Kreidieh, K. Parvate, E. Vinitsky, A. Bayen, "Flow: Architecture and Benchmarking for Reinforcement Learning in Traffic Control," CoRR, vol. abs/1710.05465, 2017. [Online]. Available: https://arxiv.org/abs/1710.05465

If you use the benchmarks, you are highly encouraged to cite our paper:

Vinitsky, E., Kreidieh, A., Le Flem, L., Kheterpal, N., Jang, K., Wu, F., ... & Bayen, A. M,  Benchmarks for reinforcement learning in mixed-autonomy traffic. In Conference on Robot Learning (pp. 399-409). Available: http://proceedings.mlr.press/v87/vinitsky18a.html

## Contributors to Original Repository

Flow is supported by the [Mobile Sensing Lab](http://bayen.eecs.berkeley.edu/) at UC Berkeley and Amazon AWS Machine Learning research grants. The contributors are listed in [Flow Team Page](https://flow-project.github.io/team.html).
