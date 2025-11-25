#!/bin/bash
### Job name
#SBATCH -J Run_Main_NoDefense
#SBATCH -o /home/hice1/khom9/CSE-8803-MLG/biomedical-gnn-privacy/membership_inference_attack/main_nodefense.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=khom9@gatech.edu
### Queue name
### Specify the number of nodes and thread (ppn) for your job.
#SBATCH -N1 --ntasks=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1 -C HX00
### Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
#SBATCH -t 09:00:00
#################################
module load pytorch
nvidia-smi
# PROTEINS
python -u run.py --dataset MUTAG --iter 15

# MUTAG
python -u run.py --dataset PROTEINS --iter 15