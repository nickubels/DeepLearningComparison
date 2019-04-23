#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=DLtrain
#SBATCH --mem=3000
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=n.s.ubels@student.rug.nl
#SBATCH --output=logs/job-%j.log

module load Python/3.6.4-foss-2018a
source venv/bin/activate
python3 script.py --epochs 1 -j $SLURM_JOB_ID