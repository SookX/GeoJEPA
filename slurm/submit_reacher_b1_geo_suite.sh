#!/bin/bash
# Submit B1+anisotropy, B1+scale, and B1+full-geo training jobs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
cd "${PROJECT_DIR}"

COMMON_EXPORT="PROJECT_DIR=${PROJECT_DIR},GEO_K=4,GEO_ALPHA0=1.0"
SCRIPT="slurm/discoverer_reacher_b1_geo_train.sh"

jid_aniso=$(sbatch --parsable \
    --export=ALL,${COMMON_EXPORT},ABLATION_NAME=b1_geo_aniso,OUTPUT_MODEL_NAME=lewm_reacher_b1_geo_aniso,RUN_ID=reacher_workshop/b1_geo_aniso_seed3072,GEO_ANISO_WEIGHT=1e-4,GEO_SCALE_WEIGHT=0 \
    "${SCRIPT}")
echo "b1_geo_aniso train job ${jid_aniso}"

jid_scale=$(sbatch --parsable \
    --export=ALL,${COMMON_EXPORT},ABLATION_NAME=b1_geo_scale,OUTPUT_MODEL_NAME=lewm_reacher_b1_geo_scale,RUN_ID=reacher_workshop/b1_geo_scale_seed3072,GEO_ANISO_WEIGHT=0,GEO_SCALE_WEIGHT=1e-3 \
    "${SCRIPT}")
echo "b1_geo_scale train job ${jid_scale}"

jid_full=$(sbatch --parsable \
    --export=ALL,${COMMON_EXPORT},ABLATION_NAME=b1_geo_full,OUTPUT_MODEL_NAME=lewm_reacher_b1_geo_full,RUN_ID=reacher_workshop/b1_geo_full_seed3072,GEO_ANISO_WEIGHT=1e-4,GEO_SCALE_WEIGHT=1e-3 \
    "${SCRIPT}")
echo "b1_geo_full train job ${jid_full}"

if [ "${SUBMIT_EVAL:-1}" = "1" ]; then
    eval_script="slurm/discoverer_reacher_eval200_loop.sh"
    jid_eval_aniso=$(sbatch --parsable --dependency=afterok:${jid_aniso} \
        --export=ALL,POLICY_TAG=b1_geo_aniso,POLICY=lewm_reacher_b1_geo_aniso/weights_epoch_3.pt,FORCE=1 \
        "${eval_script}")
    echo "b1_geo_aniso eval job ${jid_eval_aniso}"

    jid_eval_scale=$(sbatch --parsable --dependency=afterok:${jid_scale} \
        --export=ALL,POLICY_TAG=b1_geo_scale,POLICY=lewm_reacher_b1_geo_scale/weights_epoch_3.pt,FORCE=1 \
        "${eval_script}")
    echo "b1_geo_scale eval job ${jid_eval_scale}"

    jid_eval_full=$(sbatch --parsable --dependency=afterok:${jid_full} \
        --export=ALL,POLICY_TAG=b1_geo_full,POLICY=lewm_reacher_b1_geo_full/weights_epoch_3.pt,FORCE=1 \
        "${eval_script}")
    echo "b1_geo_full eval job ${jid_eval_full}"
fi
