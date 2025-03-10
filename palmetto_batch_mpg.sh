#!/bin/bash

# Submit this script with: sbatch FILENAME

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 2    # number of cpu cores per task
#SBATCH --time 16:00:00      # walltime
#SBATCH --mem 16gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task v100s:2 # gpu model and amount requested
#SBATCH --job-name "rubric-grading-workbench-mpg" # job name
# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_mem=32gb&use_gpus=yes&gpu_model=a100&walltime=10%3A00%3A00&job_name=rubric-grading-workbench

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source .venv/bin/activate
export BATCH_SIZE=10
export GPU_DEVICE=0
export NUM_WORKERS=2
sh walkthrough-dl20_phases2thru4_mpg.sh 2>&1 |tee palmetto_batch_mpg_run.log
