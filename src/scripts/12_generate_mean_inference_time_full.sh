#/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

python3 ../12_generate_mean_inference_time_full.py 
       