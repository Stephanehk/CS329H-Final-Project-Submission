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

python3 -m utils.pandemic_rollout_and_save --start_idx 0 --end_idx 1 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 1 --end_idx 2 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 2 --end_idx 3 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 3 --end_idx 4 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 4 --end_idx 5 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 5 --end_idx 6 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 6 --end_idx 7 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 7 --end_idx 8 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 8 --end_idx 9 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 9 --end_idx 10 --town_sz medium
