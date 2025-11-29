#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next6
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=pandemic_rew_learning
#SBATCH --mem=28G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=../run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/llm_reward_design/
conda activate orpo

# python3 -m reward_learning.rewards_dictionary_eval --env_name pandemic --weights_name pandemic_o4-mini_True_prefs
python3 -m reward_learning.rewards_dictionary_eval --env_name pandemic --no-convo-base-line --weights_name pandemic_no_convo_baseline_o4-mini_True_prefs

# python3 -m reward_learning.offline_policy_eval --env_name pandemic --weights_name pandemic_o4-mini_True_prefs

# python3 -m reward_learning.rewards_dictionary_eval --env_name pandemic --weights_name pandemic_healthy_extreme_o4-mini_True_prefs


#python3 -m reward_learning.offline_policy_eval --env_name pandemic --weights_name pandemic_o4-mini_True_prefs
