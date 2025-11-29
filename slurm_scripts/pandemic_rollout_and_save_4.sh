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

python3 -m utils.pandemic_rollout_and_save --start_idx 30 --end_idx 31 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 31 --end_idx 32 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 32 --end_idx 33 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 33 --end_idx 34 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 34 --end_idx 35 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 35 --end_idx 36 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 36 --end_idx 37 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 37 --end_idx 38 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 38 --end_idx 39 --town_sz medium
python3 -m utils.pandemic_rollout_and_save --start_idx 39 --end_idx 40 --town_sz medium