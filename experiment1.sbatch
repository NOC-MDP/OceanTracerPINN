#!/bin/bash
#SBATCH --job-name=OceanTracerPINN_train
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#
micromamba activate AISIT2
python train.py --config config/experiments/train1.yml
