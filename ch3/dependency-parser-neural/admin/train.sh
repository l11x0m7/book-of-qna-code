#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
dynetSeed=123321889
dynetMem=1024 # MB
outdir=$baseDir/../model
trainData=/tools/conllu-data/evsam05/THU/train.conllu
testData=/tools/conllu-data/evsam05/THU/dev.conllu
epochs=30
lr=0.01
lstmdims=100
lstmlayers=2
embedding=/tools/words.vector.txt

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/..
python app/parser.py \
    --dynet-seed $dynetSeed \
    --dynet-mem $dynetMem \
    --outdir $outdir \
    --train $trainData \
    --dev $testData \
    --epochs $epochs \
    --lstmdims $lstmdims \
    --lstmlayers $lstmlayers \
    --extrn $embedding \
    --lr $lr \
    --bibi-lstm \
    --userlmost
