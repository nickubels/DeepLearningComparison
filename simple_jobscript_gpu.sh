#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=DLtrain
#SBATCH --mem=3000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=n.s.ubels@student.rug.nl
#SBATCH --output=logs/job-%j.log

module load Python/3.6.4-foss-2018a
module load CUDA/9.1.85
source venv/bin/activate
python3 script.py --epochs 40 --gpu -j $SLURM_JOB_ID
