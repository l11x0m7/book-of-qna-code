#!/usr/bin/python
# Copyright 2010 Yoav Goldberg
##
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
##
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
##
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.
"""
Author: Yoav Goldberg
"""
from __future__ import print_function
from __future__ import division

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curdir)

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

from absl import app
from absl import flags
from absl import logging

import yutils
from collections import defaultdict
import common


def to_tok(line):
    # if line[4]=="_": line[4]=line[3]
    return {"parent": int(line[6]),
            "prel": line[7],
            "form": line[1],
            "lem": line[2],
            "id": int(line[0]),
            "tag": line[4],
            "ctag": line[3],
            "morph": line[5].split("|"),
            "extra": line[9],
            }


def conll_to_sents(fh, ignore_errs=True):
    for sent in yutils.tokenize_blanks(fh):
        if ignore_errs and sent[0][0][0] == "@":
            continue
        lines = []
        for x in sent:
            if x[0].strip().startswith("#"):
                continue
            if x[6].strip() == "_" or x[7].strip() == "_":
                continue
            if len(x) != 10:
                continue
            lines.append(x)
        if len(lines) > 0:
            yield [to_tok(l) for l in lines]


def ann_conll_to_sents(fh):
    sent = []
    for line in fh:
        line = line.strip()
        if not line:
            if sent:
                yield [to_tok(l) for l in sent]
                sent = []
        elif line.startswith("@@"):
            if sent:
                yield sent
                sent = []
            yield line
        else:
            sent.append(line.split())
    if sent:
        yield [to_tok(l) for l in sent]


def conll_to_sents2(fh, ignore_errs=True):
    from common import ROOT
    for sent in yutils.tokenize_blanks(fh):
        if ignore_errs and sent[0][0][0] == "@":
            continue
        sent = [to_tok(l) for l in sent]
        for tok in sent:
            par = tok['parent']
            if par == 0:
                tok['partok'] = ROOT
            elif par == -1:
                tok['partok'] = None
            else:
                tok['partok'] = sent[par - 1]
        yield sent


def read_dep_trees(fh, ignore_errs=True):
    for sent in conll_to_sents(fh, ignore_errs):
        yield DepTree(sent)


def kbest_conll_to_sents(fh, ignore_errs=True):
    while True:
        count = int(fh.next().strip())
        k = []
        for i, sent in zip(xrange(count), yutils.tokenize_blanks(fh)):
            if ignore_errs and sent[0][0][0] == "@":
                continue
            k.append([to_tok(l) for l in sent])
        yield k


class DepTree:
    def __init__(self, sent):
        self.toks = sent[:]
        self._tok_by_id = dict([(t['id'], t) for t in sent])
        self._tok_by_id[0] = common.ROOT
        self._childs = defaultdict(list)
        self._parents = {}
        for tok in sent:
            self._childs[tok['parent']].append(tok)
            self._parents[tok['id']] = self._tok_by_id[tok['parent']]

    def itertokens(self):
        for t in self.toks:
            yield t

    def parent(self, tok):
        return self._parents[tok['id']]

    def childs(self, tok):
        return self._childs[tok['id']]


def out_conll(sent, out=sys.stdout, parent='parent', form='form'):
    for tok in sent:
        out.write("%s\n" % " ".join(map(str,
                                        [tok['id'],
                                         tok[form],
                                            "_",
                                            tok['tag'],
                                            tok['tag'],
                                            "_",
                                            tok[parent],
                                            tok['prel'],
                                            "_",
                                            tok.get('extra',
                                                    '_')])))
    out.write("\n")


def add_parents_annotation(sents, parents_file):
    sents = list(sents)
    parents_annotations = list(yutils.tokenize_blanks(file(parents_file)))
    assert len(sents) == len(parents_annotations)
    for s, p in zip(sents, parents_annotations):
        assert len(s) == len(p)
        for tok, parents in zip(s, p):
            id = int(parents[0])
            pars = [int(x.split(":")[0]) for x in parents[1:]]
            scrs = [x.split(":")[1] for x in parents[1:]]
            assert(id == tok['id'])
            tok['cand_parents'] = pars
    return sents

def transform_conll_sents(conll_file_path, only_projective = False, unlex = False):
    '''
    Transform CoNLL data as feeding
    '''
    sents = list(conll_to_sents(open(conll_file_path, "r").readlines()))

    if only_projective:
        sents = [s for s in sents if is_projective(s)]

    if unlex:
        from shared.lemmatize import EnglishMinimalWordSmoother
        smoother = EnglishMinimalWordSmoother.from_words_file("1000words")
        for sent in sents:
            for tok in sent:
                tok['oform'] = tok['form']
                tok['form'] = smoother.get(tok['form'])

    return sents

