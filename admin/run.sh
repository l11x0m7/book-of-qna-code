#! /bin/bash 
###########################################
#
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
cd $baseDir/..
echo "拉取最新的docker镜像 ..."
docker pull chatopera/qna-book:1.0.1
echo "运行容器 ..."
# -p 9200:9200 \
# -p 9300:9300 \
docker run -it --rm \
	-v $PWD:/app \
    -v $PWD/ch3/search-engine/data:/usr/share/elasticsearch/data \
    -v $PWD/ch3/search-engine/config:/usr/share/elasticsearch/config \
    -v $PWD/ch3/search-engine/plugins:/usr/share/elasticsearch/plugins \
	chatopera/qna-book:1.0.1 \
	bash

