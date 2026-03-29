#!/bin/bash

cd ../..

DATA=/gemini/code/RPCW/data/

DATASET=$1
NLAB=$2 # total number of labels
CFG=$3 # v1, v2, v3, v4

if [ ${DATASET} == ssdg_pacs ]; then
    # NLAB: 210 or 105
    D1=art_painting
    D2=cartoon
    D3=photo
    D4=sketch
elif [ ${DATASET} == ssdg_vlcs ]; then
    # NLAB: 150 or 75
    D1=CALTECH
    D2=LABELME
    D3=PASCAL
    D4=SUN
fi

TRAINER=RPCW
NET=resnet34

for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

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
done