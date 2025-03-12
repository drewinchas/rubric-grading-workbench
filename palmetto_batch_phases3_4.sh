#!/bin/bash

# Submit this script with: sbatch FILENAME

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 1    # number of cpu cores per task
#SBATCH --time 1:00:00      # walltime
#SBATCH --mem 8gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --job-name "rubric-grading-phases3_4-mp" # job name
# Created with the RCD Docs Job Builder
#

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source .venv/bin/activate
export BATCH_SIZE=6
export GPU_DEVICE=0
export NUM_WORKERS=2
sh walkthrough-dl20_phases3thru4.sh 2>&1 |tee palmetto_phases3_4_run.log
