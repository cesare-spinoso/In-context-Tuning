#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=6                                # Ask for 6 CPUs
#SBATCH --gres=gpu:a100l:1                                    # Ask for 1 GPU but an a100l (should have 80G of gpu memory)
#SBATCH --time=24:00:00                                  # The job will run for 3 hours
#SBATCH --output=results/biclfs_training_job_output.txt
#SBATCH --error=results/biclfs_training_job_error.txt

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate ict

# 3. Launch
python src/biclfs_training.py
