import os
import time
import subprocess

# Directory where the SLURM job scripts are located
job_dir = "traffic_gt_rew_set_runs"

# Collect all .sh files in the directory, sorted by index
job_files = sorted(
    [f for f in os.listdir(job_dir) if f.endswith(".sh")],
    key=lambda x: int(x.split("_")[-1].split(".")[0])  # extract number from filename
)

# Submit each job with a delay
for job_file in job_files:
    job_path = os.path.join(job_dir, job_file)
    print(f"Submitting {job_path} ...")
    subprocess.run(["sbatch", job_path])
    time.sleep(5)  # wait 10 seconds between submissions

print("All jobs submitted.")