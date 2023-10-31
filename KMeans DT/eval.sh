#!/bin/bash
DATA_PATH=$1 
K=$2
ENCODER_TYPE=$3
DISTANCE=$4

ENCODER_TYPES=('resnet' , 'vit')
DISTANCES=('cosine' , 'euclidean' , 'manhattan')


if [ -z "$ENCODER_TYPE" ]; then
    ENCODER_TYPE='resnet'
fi

# Check if DISTANCE is missing
if [ -z "$DISTANCE" ]; then
    DISTANCE='manhattan'
fi

# Check if K is missing
if [ -z "$K" ]; then
    K=12
fi

python -c "from test import main_test; main_test('${DATA_PATH}'  , k = ${K}, encoder_type = '${ENCODER_TYPE}' , distance = '${DISTANCE}'); "