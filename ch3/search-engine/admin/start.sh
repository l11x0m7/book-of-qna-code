#! /bin/bash 
###########################################
# Start elasticsearch script
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
ELASTICSEARCH_HOME=/usr/share/elasticsearch
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $ELASTICSEARCH_HOME
echo "start elasticsearch version:" `bin/elasticsearch -version`
bin/elasticsearch -Des.insecure.allow.root=true

