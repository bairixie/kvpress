#!/bin/bash

# Script to sequentially run SVD baseline and SVD lowrank methods on a single GPU
# This ensures fair comparison by running them one after another on the same hardware

dataset="ruler"
data_dir="4096"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"
compression_ratios=(0.5 0.9)
device="cuda:0"  # Run all methods on the same GPU

# Press names to compare
press_names=("svd_baseline" "svd_lowrank")

echo "Starting SVD comparison evaluation on $device"
echo "Dataset: $dataset, Data dir: $data_dir, Model: $model"
echo "Press methods: ${press_names[@]}"
echo "Compression ratios: ${compression_ratios[@]}"
echo "=========================================="

# Iterate over press names sequentially (not in parallel)
for press in "${press_names[@]}"; do
  echo ""
  echo "=========================================="
  echo "Starting evaluation for: $press"
  echo "=========================================="
  
  # Iterate over compression ratios for this press
  for compression_ratio in "${compression_ratios[@]}"; do
    echo ""
    echo "Running $press with compression_ratio: $compression_ratio on $device"
    echo "----------------------------------------"
    
    python evaluate.py \
      --dataset "$dataset" \
      --data_dir "$data_dir" \
      --model "$model" \
      --press_name "$press" \
      --compression_ratio "$compression_ratio" \
      --device "$device"
    
    # Check if the command succeeded
    if [ $? -ne 0 ]; then
      echo "Error: Failed to run $press with compression_ratio $compression_ratio"
      exit 1
    fi
    
    echo "Completed: $press with compression_ratio: $compression_ratio"
  done
  
  echo ""
  echo "=========================================="
  echo "Completed all compression ratios for: $press"
  echo "=========================================="
done

echo ""
echo "=========================================="
echo "All SVD comparison evaluations completed successfully!"
echo "=========================================="

