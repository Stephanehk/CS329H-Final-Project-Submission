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

python3 -m utils.pandemic_rollout_and_save --start_idx 10 --end_idx 11 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 11 --end_idx 12 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 12 --end_idx 13 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 13 --end_idx 14 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 14 --end_idx 15 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 15 --end_idx 16 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 16 --end_idx 17 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 17 --end_idx 18 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 18 --end_idx 19 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 19 --end_idx 20 --town_sz medium