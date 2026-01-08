#!/bin/bash

#-----------------------------------------------------------
# Demo script: Submit multiple jobs to a SLURM queue using FDQ.
#-----------------------------------------------------------

submit_job() {
    root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 $root_path/fdq_submit.py $root_path/$1
}

submit_job experiment_templates/mnist/mnist_class_dense.yaml

submit_job experiment_templates/segment_pets/segment_pets_01.yaml
submit_job experiment_templates/segment_pets/segment_pets_02_noAMP_resubmit.yaml
submit_job experiment_templates/segment_pets/segment_pets_03_no_scratch.yaml
submit_job experiment_templates/segment_pets/segment_pets_04_distributed_w2.yaml
submit_job experiment_templates/segment_pets/segment_pets_05_distributed_w4.yaml
submit_job experiment_templates/segment_pets/segment_pets_06_cached.yaml
submit_job experiment_templates/segment_pets/segment_pets_07_cached_augmentations.yaml
submit_job experiment_templates/segment_pets/segment_pets_08_distributed_cached.yaml