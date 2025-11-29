#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next6
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=pandemic_rollout
#SBATCH --mem=28G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=../run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/llm_reward_design/
conda activate orpo

python3 -m utils.pandemic_rollout_and_save --start_idx 40 --end_idx 41 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 41 --end_idx 42 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 32 --end_idx 33 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 43 --end_idx 44 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 44 --end_idx 45 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 45 --end_idx 46 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 46 --end_idx 47 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 47 --end_idx 48 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 48 --end_idx 49 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 49 --end_idx 50 --town_sz medium