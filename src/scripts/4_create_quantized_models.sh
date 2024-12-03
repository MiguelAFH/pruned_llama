#/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

# This script is in charge of creating the qunatized models for 4-bit and 8-bit versions
# from the results of II.

python3 ../4_create_quantized_models.py 
       