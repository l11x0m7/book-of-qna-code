#! /bin/bash 
###########################################
#
###########################################

function test(){
    python $1 \
        --verbosity=$2 \
        --test=True \
        --model=$3 \
        --test_data=$4 \
        --test_results=$5
}

function train(){
    python $1 \
        --verbosity=$2 \
        --model=$3 \
        --train=True \
        --train_data=$4 \
        --epoch=$5
}
