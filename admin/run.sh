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
docker run -it --rm \
	-v $PWD:/app \
	chatopera/qna-book:1.0.1 \
	bash

