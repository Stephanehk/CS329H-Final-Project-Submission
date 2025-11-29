import os


env_name = "traffic"

# Directory to store the generated SLURM files
output_dir = f"{env_name}_gt_rew_set_runs"
os.makedirs(output_dir, exist_ok=True)

# SLURM file template
slurm_template = """#!/bin/bash
#SBATCH --partition=next
#SBATCH --account=next
#SBATCH --nodelist=next{node_i}
#SBATCH --time=200:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name={env_name}_rew_learning
#SBATCH --cpus-per-task={n_cpus}
#SBATCH --mem=16G

### Logging
#SBATCH --output=/next/u/stephhk/llm_reward_design/gt_rew_fn_run_logs/slurmjob_%j.out                    # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/next/u/stephhk/llm_reward_design/gt_rew_fn_run_logs/slurmjob_%j.err                        # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=stephhk@stanford.edu # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE

cd /next/u/stephhk/llm_reward_design/
conda activate orpo

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
exec 1>&2  # Redirect stdout (fd 1) to stderr (fd 2)

python3 -m rl_utils.train_with_custom_rew \\
    --env-type {env_name} \\
    --num-workers {n_cpus} \\
    --num-gpus 1 \\
    --num-iterations 200 \\
    --reward-fun-type gt_rew_set \\
    --gt-rew-i {rew_i} \\
    {extra_args}
"""


# n_cpus = 20 if env_name == "traffic" else 10
n_cpus=10
# Generate 50 files
for i in range(50):
    if env_name == "traffic":
        node_i = 3 if i < 25 else 4
    else:
        node_i = 6 if i < 25 else 5
    filename = os.path.join(output_dir, f"{env_name}_train_ppo_with_true_rew_{i}.sh")
    if i >= 25:
        extra_args = "--flip-sign"
    else:
        extra_args = ""
    with open(filename, "w") as f:
        f.write(slurm_template.format(rew_i=i, node_i=node_i, extra_args = extra_args, env_name=env_name,n_cpus=n_cpus))

print(f"Generated 50 SLURM job files in '{output_dir}'")
