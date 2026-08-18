#!/bin/bash
#SBATCH --job-name=OceanTracerPINN_infer
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#
export PYTHONUNBUFFERED=1
micromamba activate AISIT2
python infer.py
