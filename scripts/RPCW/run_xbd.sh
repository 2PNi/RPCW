#!/bin/bash

cd ../..

DATA=/gemini/code/RPCW/data/

DATASET=$1
NLAB=$2 # total number of labels
CFG=$3 # v1, v2, v3, ...

if [ ${DATASET} == ssdg_xbd ]; then
    D1=d1
    D2=d2
fi

TRAINER=RPCW
NET=resnet34

for SEED in $(seq 1 5)
do
    S1=${D1}
    T=${D2}

    python train_rpcw.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --source-domains ${S1} ${S2} ${S3} \
    --target-domains ${T} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${DATASET}_${CFG}.yaml \
    --output-dir output/${DATASET}/nlab_${NLAB}/${TRAINER}/${NET}/${CFG}/${T}/seed${SEED} \
    MODEL.BACKBONE.NAME ${NET} \
    DATASET.NUM_LABELED ${NLAB}
done