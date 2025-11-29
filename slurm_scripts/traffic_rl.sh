#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next7
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=traffic_rew_learning
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G

### Logging
#SBATCH --output=../run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=../run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/llm_reward_design/
conda activate orpo

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec 1>&2  # Redirect stdout (fd 1) to stderr (fd 2)

# python3 -m rl_utils.train_with_custom_rew --env-type traffic --num-workers 10 --num-gpus 1 --num-iterations 500 --init-checkpoint
python3 -m rl_utils.train_with_custom_rew --env-type traffic --no-convo-base-line --reward-fun-type learned_rew --num-workers 10 --num-gpus 1 --num-iterations 500 
#--init-checkpoint