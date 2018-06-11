#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===============================================================================
#
# Copyright (c) 2017 <> All Rights Reserved
#
#
# Author: Hai Liang Wang
# Date: 2018-06-11:16:41:22
#
#===============================================================================

"""
   
"""
from __future__ import print_function
from __future__ import division

__copyright__ = "Copyright (c) 2017 . All Rights Reserved"
__author__    = "Hai Liang Wang"
__date__      = "2018-06-11:16:41:22"


import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"
else:
    unicode = str
    raise "Must be using Python 2"

# Get ENV
ENVIRON = os.environ.copy()

import jieba
import lucene
import glob
import subprocess
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser

try:
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
except Exception, e:
    print(e)

TEXT_DIR = os.path.join(curdir, os.path.pardir, "data")
INDEX_DIR = os.path.join(curdir, os.path.pardir, "indexes", "workarea")
ANALYZER = WhitespaceAnalyzer()


def exec_cmd(cmd):
    '''
    exec a string as shell scripts
    return
    '''
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=ENVIRON)
    out, err = p.communicate()
    return out, err

def create_dir(target, remove = False):
    '''
    Create a folder.
    return: stdout, stderr
    '''
    rmstr = ""
    if os.path.exists(target):
        if remove: rmstr = "rm -rf %s &&" % target
        else: 
            return None
    return exec_cmd('%smkdir -p %s' % (rmstr, target))

def get_terms(filepath):
    '''
    Get terms for postings
    '''
    with open(filepath, 'r') as fin:
        lines = list(fin.readlines())
        doc = " ".join(lines) 
    serial, abstract = filepath.split(os.sep)[-1].split('.')[:-1]
    title = "第%s回 %s" % (serial, abstract)
    print(">> 建立倒排索引：%s" % title)
    terms = " ".join(jieba.cut(doc, cut_all=False))

    # print("filepath: %s, content: %s" % (filepath, doc))
    return title, doc, terms

'''
索引数据
'''
def indexing():
    print("建立索引，文本文件夹 [%s] ..." % TEXT_DIR)
    create_dir(INDEX_DIR)
    directory = SimpleFSDirectory(Paths.get(INDEX_DIR))
    config = IndexWriterConfig(ANALYZER)
    writer = IndexWriter(directory, config)

    for x in glob.glob(os.path.join(TEXT_DIR, "*.txt")):
        title, post, terms = get_terms(x)
        doc = Document()
        if terms:
            doc.add(Field("title", title, TextField.TYPE_STORED))
            doc.add(Field("post", post, TextField.TYPE_STORED))
            doc.add(Field("terms", terms, TextField.TYPE_STORED))
            writer.addDocument(doc)

    writer.commit()
    writer.close()

# 如果索引文件夹不存在，就建立索引
if not os.path.exists(INDEX_DIR):
    indexing()


'''
查询文档
'''
def query(q, size = 20):
    print("\n查询文档： ", q)
    query = QueryParser("terms", ANALYZER).parse(q)
    directory = SimpleFSDirectory(Paths.get(INDEX_DIR))
    searcher = IndexSearcher(DirectoryReader.open(directory))
    scoreDocs = searcher.search(query, size).scoreDocs
    results = []
    for scoreDoc in scoreDocs:
        doc = searcher.doc(scoreDoc.doc)
        print("<< 相关文档： %s， 分数：%s" % (doc.get("title"), scoreDoc.score))
        results.append(dict({
                            "post": doc.get("post"),
                            "title": doc.get("title"),
                            "score": scoreDoc.score,
                            "terms": doc.get("terms")
                            }))
    return results

# 执行查询
query("诸葛亮 司马懿 -吕蒙")

# if __name__ == '__main__':
#     print("pass")