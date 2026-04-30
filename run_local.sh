#!/bin/bash

#-----------------------------------------------------------
# Demo script: Run multiple jobs locally (sequentially)
#-----------------------------------------------------------
root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

function submit_job() {
    local config_path=$1
    local config_name=$2
    echo "----------------------------------------"
    echo "Starting job with config: $config_name"
    fdq --config-path "$config_path" --config-name "$config_name"
    echo "Finished job with config: $config_name"
    echo "----------------------------------------"
}

mnist_path="$root_path/experiment_templates/mnist"
submit_job "$mnist_path" mnist_class_dense

oxford_pets_path="$root_path/experiment_templates/segment_pets"
submit_job "$oxford_pets_path" segment_pets_01
submit_job "$oxford_pets_path" segment_pets_02_noAMP_resubmit
submit_job "$oxford_pets_path" segment_pets_03_no_scratch

