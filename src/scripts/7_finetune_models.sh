#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

cd ../../qlora

# Make sure to use the qlora python environment

CUDA_VISIBLE_DEVICES=0 python qlora.py \
    --model_name_or_path /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.1/ \
    --bits 4 \
    --output_dir /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.1-4bit-qlora/ &

CUDA_VISIBLE_DEVICES=1 python qlora.py \
    --model_name_or_path /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.2/ \
    --bits 4 \
    --output_dir /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.2-4bit-qlora/ &

CUDA_VISIBLE_DEVICES=2 python qlora.py \
    --model_name_or_path /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.3/ \
    --bits 4 \
    --output_dir /share/pi/nigam/users/migufuen/temp/qlora/models/Llama-3.2-1B-Instruct_pruned_0.3-4bit-qlora/ &

wait