#!/bin/bash
#SBATCH --job-name="Inventory_calc"
#SBATCH --time=4:00:00
#SBATCH --mem=200Gb
#SBATCH --account=nemo
#SBATCH --partition=standard
#SBATCH --qos=short

micromamba activate AISIT2

python create_inventory.py
