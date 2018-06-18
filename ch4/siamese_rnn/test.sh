#! /bin/bash 
###########################################
#  Train Model
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
source ~/venv-py2/bin/activate # Use python2
export GLOVE_EMBEDDING_6B=/tools/embedding/glove.6B.100d.txt

# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir
echo "test model"
python siamese_nn.py --test