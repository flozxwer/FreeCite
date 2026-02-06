#!/bin/bash

# Define the list of model names
models=("")

# Loop through each model and run the command in the background
for model in "${models[@]}"; do
    python citation_try_code_eng_data.py --model_name "$model" &
done

# Wait for all background processes to finish
wait

echo "All processes have completed."