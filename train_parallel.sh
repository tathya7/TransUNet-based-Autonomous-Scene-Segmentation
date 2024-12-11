#!/bin/bash

# This script runs the training of the UNet, TransUNet, and SwinUNet models in parallel.
# The script will automatically assign the models to the GPUs with the least memory usage.
# The training logs will be saved in the `logs` directory.

check_gpu() {
    nvidia-smi -i $1 &> /dev/null
    return $?
}

get_gpu_memory_usage() {
    local gpu_id=$1
    local memory_usage=$(nvidia-smi --id=$gpu_id --query-gpu=memory.used --format=csv,noheader,nounits)
    echo $memory_usage
}


mkdir -p logs/unet
mkdir -p logs/transunet
mkdir -p logs/swinunet
+x train_parallel.sh

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Found $NUM_GPUS GPUs"

if [ $NUM_GPUS -lt 3 ]; then
    echo "Warning: Less than 3 GPUs available. Some models will run on the same GPU."
fi


declare -A gpu_usage
for ((i=0; i<$NUM_GPUS; i++)); do
    if check_gpu $i; then
        memory_usage=$(get_gpu_memory_usage $i)
        gpu_usage[$i]=$memory_usage
    fi
done


IFS=$'\n' sorted_gpus=($(for gpu in "${!gpu_usage[@]}"; do 
    echo "$gpu ${gpu_usage[$gpu]}"
done | sort -k2n))
unset IFS


GPU_UNET=${sorted_gpus[0]%% *}
GPU_TRANSUNET=${sorted_gpus[1]%% *}
GPU_SWINUNET=${sorted_gpus[2]%% *}


if [ $NUM_GPUS -lt 3 ]; then
    GPU_SWINUNET=$GPU_UNET
fi
if [ $NUM_GPUS -lt 2 ]; then
    GPU_TRANSUNET=$GPU_UNET
fi

echo "GPU Assignments:"
echo "UNet: GPU $GPU_UNET"
echo "TransUNet: GPU $GPU_TRANSUNET"
echo "SwinUNet: GPU $GPU_SWINUNET"


CUDA_VISIBLE_DEVICES=$GPU_UNET python train/unet_train.py \
    --save-dir "unet_results" > logs/unet/training.log 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU_TRANSUNET python train/transunet_train.py \
    --save-dir "transunet_results" > logs/transunet/training.log 2>&1 &

CUDA_VISIBLE_DEVICES=$GPU_SWINUNET python train/swinunet_train.py \
    --save-dir "swinunet_results" > logs/swinunet/training.log 2>&1 &

wait

echo "All training processes completed!"


for job in $(jobs -p); do
    wait $job || let "FAIL+=1"
done

if [ "$FAIL" == "0" ]; then
    echo "All training jobs completed successfully!"
else
    echo "Some training jobs failed. Check the logs for details."
fi
