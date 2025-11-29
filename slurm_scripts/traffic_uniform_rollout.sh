#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next7
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=traffic_rollout
#SBATCH --mem=28G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=../run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/llm_reward_design/
conda activate orpo

python3 -m rl_utils.rollout_uniform_policy --env-type traffic --num-workers 10 --num-gpus 1 
#--init-checkpoint