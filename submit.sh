#!/bin/bash

#-----------------------------------------------------------
# Demo script: Submit multiple jobs to a SLURM queue using FDQ.
#-----------------------------------------------------------

submit_job() {
    root_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if command -v fdq_submit >/dev/null 2>&1; then
        # this only works if fdq was installed (pip install fdq[submit])
        fdq_submit "$root_path/$1"
    else
        # alternatively, you can run the submit module directly (without installing fdq)
        PYTHONPATH="$root_path/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m fdq.submit "$root_path/$1"
    fi
}
# Train an MNIST classifier using a simple dense architecture, and do a small param study on the learning rate and batch size.
submit_job experiment_templates/mnist/mnist_class_dense.yaml
submit_job experiment_templates/mnist/mnist_class_dense_param_study.yaml


# Train OXFORD Pets segmentation using a simple (oversized) UNET architecture.
# Show benefit of DDP using 2 and 4 GPUs.
submit_job experiment_templates/segment_pets/segment_pets_01.yaml
submit_job experiment_templates/segment_pets/segment_pets_02_dist2.yaml
submit_job experiment_templates/segment_pets/segment_pets_03_dist4.yaml


# Train OXFORD Pets segmentation using the Chuchichaestli UNET architecture.
# Show automatic job resubmission and dataset caching.
submit_job experiment_templates/segment_pets/segment_pets_10.yaml
submit_job experiment_templates/segment_pets/segment_pets_11_noAMP_resubmit.yaml
submit_job experiment_templates/segment_pets/segment_pets_12_slow.yaml
submit_job experiment_templates/segment_pets/segment_pets_13_cached.yaml
submit_job experiment_templates/segment_pets/segment_pets_14_cached_augmentations.yaml
