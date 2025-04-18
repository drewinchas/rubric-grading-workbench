#!/bin/bash

# Submit this script with: sbatch FILENAME

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 2    # number of cpu cores per task
#SBATCH --time 16:00:00      # walltime
#SBATCH --mem 16gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task v100s:2 # gpu model and amount requested
#SBATCH --job-name "rubric-grading-workbench-flan-t5-large" # job name
# Created with the RCD Docs Job Builder
#

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source .venv/bin/activate
export BATCH_SIZE=10
export GPU_DEVICE=0
export NUM_WORKERS=2
sh walkthrough-dl20_phases2thru4_flan-t5-large.sh 2>&1 |tee palmetto_batch_flan-t5-large_run.log
