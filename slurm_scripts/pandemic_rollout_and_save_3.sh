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

python3 -m utils.pandemic_rollout_and_save --start_idx 20 --end_idx 21 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 21 --end_idx 22 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 22 --end_idx 23 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 23 --end_idx 24 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 24 --end_idx 25 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 25 --end_idx 26 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 26 --end_idx 27 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 27 --end_idx 28 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 28 --end_idx 29 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 29 --end_idx 30 --town_sz medium
