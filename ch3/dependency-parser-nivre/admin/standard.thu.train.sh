#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
. $baseDir/util.sh


#######################
# variables
#######################
PY=$baseDir/../app/transitionparser/standard.py
TRAIN_DATA=/tools/conllu-data/evsam05/THU/train.conllu
MODEL=$baseDir/../model/standard.thu.model
EPOCH=10
LOG_VERBOSITY=0 # info

# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
set -x
cd $baseDir/..
if [ ! -d model ]; then
    mkdir model
fi
train $PY $LOG_VERBOSITY $MODEL $TRAIN_DATA $EPOCH 
