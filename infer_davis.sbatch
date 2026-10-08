#!/bin/bash
#SBATCH --job-name=OceanTracerPINN_infer
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#
export PYTHONUNBUFFERED=1
micromamba activate AISIT2

# Define year range
START_YEAR=2005
END_YEAR=2023

# Loop over all years and months
for YEAR in $(seq $START_YEAR $END_YEAR); do
    for MONTH in $(seq -w 1 12); do
        echo "Running inference for Year: $YEAR, Month: $MONTH"
        python infer.py \
            --config config/inference/infer_davis.yml \
            --year $YEAR \
            --month $MONTH \
        # Check if the python command succeeded
        if [ $? -ne 0 ]; then
            echo "ERROR: Inference failed for Year: $YEAR, Month: $MONTH — skipping."
        fi
    done
done
