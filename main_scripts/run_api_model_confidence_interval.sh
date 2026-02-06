#!/bin/bash

# Define the model name
model_name=""

# Define different seed values
seeds=(10 20 30 40 50)

# Loop through the seeds and run the script
for seed in "${seeds[@]}"; do
    if ! python api_large_experiment_confidence_interval.py --seed "$seed" --model_name "$model_name"; then
        echo "Error occurred with seed $seed"
        exit 1
    fi
done