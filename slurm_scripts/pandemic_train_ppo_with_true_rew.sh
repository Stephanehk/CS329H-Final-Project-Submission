#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next7
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=glucose_rew_learning
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

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


# python -m occupancy_measures.experiments.orpo_experiments with env_to_run=pandemic reward_fun=true exp_algo=ORPO 'om_divergence_coeffs=['0.0']' 'checkpoint_to_load_policies=None' 'checkpoint_to_load_current_policy=None' seed=0 experiment_tag=state-action num_training_iters=0 num_gpus=1 num_rollout_workers=10

python3 -m rl_utils.train_with_custom_rew --env-type pandemic --num-workers 10 --num-gpus 1 --num-iterations 100 --reward-fun-type gt_rew_set --gt-rew-i 0