#!/bin/bash

# This script is in charge of evaluating the performance
# of the pruned models from I.
# /share/pi/nigam/users/migufuen/helm/src/helm/config/model_deployments.yaml

export SUITE_NAME=PrunedMedLLM
export RUN_ENTRIES_CONF_PATH=2_evaluate_pruned_models.conf
export SCHEMA_PATH=/share/pi/nigam/users/migufuen/temp/pruned_llama/helm/src/helm/benchmark/static/schema_medical.yaml
export NUM_TRAIN_TRIALS=1
export MAX_EVAL_INSTANCES=1000
export PRIORITY=2

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd $SCRIPT_DIR

helm-run --conf-paths $RUN_ENTRIES_CONF_PATH --num-train-trials $NUM_TRAIN_TRIALS --max-eval-instances $MAX_EVAL_INSTANCES --priority $PRIORITY --suite $SUITE_NAME

helm-summarize --schema $SCHEMA_PATH --suite $SUITE_NAME
