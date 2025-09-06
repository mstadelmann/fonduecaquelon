#!/bin/bash

#-----------------------------------------------------------
# Demo script: Submit multiple jobs to a SLURM queue using FDQ.
#-----------------------------------------------------------

submit_job() {
    root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 $root_path/fdq_submit.py $root_path/$1
}

submit_job experiment_templates/mnist/mnist_class_dense.json

submit_job experiment_templates/segment_pets/segment_pets_01.json
submit_job experiment_templates/segment_pets/segment_pets_01_noAMP.json
submit_job experiment_templates/segment_pets/segment_pets_02_no_scratch.json
submit_job experiment_templates/segment_pets/segment_pets_03_distributed_w2.json
submit_job experiment_templates/segment_pets/segment_pets_04_distributed_w4.json
submit_job experiment_templates/segment_pets/segment_pets_05_cached.json
submit_job experiment_templates/segment_pets/segment_pets_06_cached_augmentations.json
submit_job experiment_templates/segment_pets/segment_pets_07_distributed_cached.json