#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gputest
#SBATCH --gres=gpu:1
#SBATCH --job-name=DLtrain
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --mail-user=n.s.ubels@student.rug.nl
#SBATCH --output=logs/%A_%a-%j.log
#SBATCH --array=1-3,5-6,8-10

INPUTFILE=input.in
ARGS=$(cat $INPUTFILE | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)

module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
source venv/bin/activate
python3 script.py --epochs 100 --gpu -j $SLURM_JOB_ID --optimizer $ARGS
