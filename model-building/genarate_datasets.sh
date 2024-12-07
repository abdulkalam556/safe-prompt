#!/bin/bash

# Define dataset names and tokenizer configurations
datasets=("ai4privacy/pii-masking-400k" "ai4privacy/pii-masking-200k")
tokenizers=("bert-base-cased" "roberta-base")
output_basenames=("Bert" "RoBERTa")

# Maximum sequence length for tokenization
max_length=512

# Process each dataset-tokenizer combination
for i in "${!tokenizers[@]}"; do
    tokenizer_name="${tokenizers[$i]}"
    output_base="${output_basenames[$i]}"
    
    for dataset_name in "${datasets[@]}"; do
        # Extract dataset size (e.g., "400k") to name the output folders correctly
        dataset_short=$(echo "$dataset_name" | grep -oP '\d+k')
        
        # Define the output path based on the specified structure
        output_path="../data/${output_base}-${dataset_short}"   # For processed datasets
        
        # Create the directory if it doesn't exist
        mkdir -p "$output_path"
        
        # Run data preparation with the specified tokenizer and save output to the target directory
        echo "Processing dataset $dataset_name with tokenizer $tokenizer_name..."
        python ../code/data_preparation.py --dataset_name "$dataset_name" \
                                           --tokenizer_name "$tokenizer_name" \
                                           --max_length "$max_length" \
                                           --output_path "$output_path"
        
        echo "Saved processed dataset to $output_path"
    done
done

echo "Dataset preparation completed for BERT and RoBERTa."