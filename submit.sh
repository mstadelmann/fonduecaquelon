#!/bin/bash

#-----------------------------------------------------------
# Demo script: Submit multiple jobs to a SLURM queue using FDQ.
#-----------------------------------------------------------

submit_job() {
    root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if command -v fdq_submit >/dev/null 2>&1; then
        fdq_submit "$root_path/$1"
    else
        PYTHONPATH="$root_path/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m fdq.submit "$root_path/$1"
    fi
}

submit_job experiment_templates/mnist/mnist_class_dense.yaml
submit_job experiment_templates/mnist/mnist_class_dense_param_study.yaml

submit_job experiment_templates/segment_pets/segment_pets_01.yaml
submit_job experiment_templates/segment_pets/segment_pets_02_noAMP_resubmit.yaml
submit_job experiment_templates/segment_pets/segment_pets_03_no_scratch.yaml
submit_job experiment_templates/segment_pets/segment_pets_04_distributed_w2.yaml
submit_job experiment_templates/segment_pets/segment_pets_05_distributed_w4.yaml
submit_job experiment_templates/segment_pets/segment_pets_06_cached.yaml
submit_job experiment_templates/segment_pets/segment_pets_07_cached_augmentations.yaml
submit_job experiment_templates/segment_pets/segment_pets_08_distributed_cached.yaml
