#/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

# This script is in charge of creating the plots
# from the results of II. It takes a df and generates
# 1 plot with a line per benchmark.  The x axis is pruning rate and y performance 
# in each benchmark

python3 ../3_generate_pruned_performance_graphs.py 
       