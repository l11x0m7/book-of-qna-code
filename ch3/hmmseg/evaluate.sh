#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions


# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir
echo "对测试文件进行分词 ..."
python eval.py
echo "评测分词结果 ..."
perl /tools/icwb2-data/scripts/score \
    /tools/icwb2-data/gold/msr_training_words.utf8 \
    /tools/icwb2-data/gold/msr_test_gold.utf8 \
    hmm.msr_test.seg \
    > hmm.msr_test.eval


echo "评测结果："
tail -n 30 hmm.msr_test.eval

echo "详情" `pwd`/hmm.msr_test.eval