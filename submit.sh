#!/bin/bash

#-----------------------------------------------------------
# Demo script: Submit multiple jobs to a SLURM queue using FDQ.
#-----------------------------------------------------------

submit_job() {
    root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python3 $root_path/fdq_submit.py $root_path/$1
}

submit_job experiment_templates/mnist/mnist_class_dense.json

submit_job experiment_templates/segment_pets/segment_pets.json
submit_job experiment_templates/segment_pets/segment_pets_distributed.json
submit_job experiment_templates/segment_pets/segment_pets_distributed_v4.json
submit_job experiment_templates/segment_pets/segment_pets_cached.json