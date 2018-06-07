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
Various feature extractors for arc-eager and arc-standard transition based parsers
Author: Yoav Goldberg
"""
from common import *
import os

# Features #{{{
__EXTRACTORS__ = {}


# extractor combinators #{{{
class ChainedFeatureExtractor:  # {{{
    def __init__(self, extSeq):
        self.extractors = extSeq

    def extract(self, stack, deps, sent, i):
        fs = []
        extend = fs.extend
        [extend(e.extract(stack, deps, sent, i)) for e in self.extractors]
        return fs
    #}}}


class AppendingFeatureExtractor:  # {{{
    def __init__(self, ext1, ext2):
        self.e1 = ext1
        self.e2 = ext2

    def extract(self, stack, deps, sent, i):
        fs = []
        fs1 = self.e1.extract(stack, deps, sent, i)
        fs2 = self.e2.extract(stack, deps, sent, i)
        fs = fs1[:]
        for f1 in fs1:
            for f2 in fs2:
                fs.append("%s^%s" % (f1, f2))
        return fs
    #}}}

#}}} end combinators


class BetterParentFeatureExtractor:  # {{{ doesn't work
    """
    this one doesn't really work very well..

    the features are "is there a better parent for X then Y,
    for example, is there a better parent for the word on top of the
    stack than the next input word?
    """

    def __init__(self, parentPredictor):
        self.parentPredictor = parentPredictor
        self.current_sent = None

    def extract(self, stack, deps, sent, i):
        if self.current_sent != sent:
            self.current_sent = sent
            self.parentPredictor.set_sent(sent)

        fs = []
        add_feature = fs.append
        w = sent[i] if len(sent) + 1 < i else PAD
        s = stack[-1] if len(stack) > 0 else PAD
        s1 = stack[-2] if len(stack) > 1 else PAD
        wparents = [p for p in self.parentPredictor.best_parents(w['id'], 3)]
        sparents = [p for p in self.parentPredictor.best_parents(s['id'], 3)]
        s1parents = [p for p in self.parentPredictor.best_parents(s1['id'], 3)]

        # is there a better parent for w then s ?
        try:
            idx = wparents.index(s['id'])
            is_better = "Y" if wparents[:idx] else "N"
        except ValueError:
            is_better = "T"
        add_feature("bp_ws_%s" % is_better)
        # is there a better parent for w than s-1 ?
        try:
            idx = wparents.index(s1['id'])
            is_better = "Y" if wparents[:idx] else "N"
        except ValueError:
            is_better = "T"
        add_feature("bp_ws1_%s" % is_better)
        # is there a better parent for s than s-1 ?
        try:
            idx = sparents.index(s1['id'])
            is_better = "Y" if sparents[:idx] else "N"
        except ValueError:
            is_better = "T"
        add_feature("bp_ss1_%s" % is_better)
        # is there a better parent for s-1 than s ?
        try:
            idx = s1parents.index(s['id'])
            is_better = "Y" if s1parents[:idx] else "N"
        except ValueError:
            is_better = "T"
        add_feature("bp_s1s_%s" % is_better)

        return fs
    #}}}


# {{{ inherit for easy graph feature extractors
class GraphBasedFeatureExtractor:
    def __init__(self, parentPredictor):
        self.pp = parentPredictor
        self.sent = None
        self.parents = {}
        self.childs = {}
        self.toks = {}

    def _init_sent(self, sent):
        if self.sent != sent:
            self.sent = sent
            self.pp.set_sent(sent)
            self.parents = {}
            self.childs = defaultdict(list)
            self.toks = {}
            for tok in sent:
                self.toks[tok['id']] = tok
            for tok in sent:
                id = tok['id']
                self.parents[id] = self.pp.best_parents(id, 3)
                for parent in self.parents[id]:
                    self.childs[parent].append(id)

    def extract(self, stack, deps, sent, i):
        self._init_sent(sent)
        return self._extract(stack, deps, sent, i)

    def _extract(self, stack, deps, sent, i):
        assert False, "must implement in child"
#}}}


class ChildsOfNextWordFeatureExtractor(GraphBasedFeatureExtractor):  # {{{
    def _extract(self, stack, deps, sent, i):
        fs = []
        if i >= len(sent):
            return fs
        w = sent[i]
        built_childs = deps.all_childs
        for child in self.childs[w['id']]:
            if child not in built_childs:
                #fs.append("w_cld_%s" % self.toks[child]['tag'])
                #fs.append("w_cld_%s" % self.toks[child]['form'])
                fs.append(
                    "st_s_cld_%s_%s" %
                    (self.toks[child]['tag'], w['tag']))
                fs.append(
                    "st_s_cld_%s_%s" %
                    (self.toks[child]['form'], w['tag']))
        return fs
    #}}}


class ChildsOfStackWordFeatureExtractor(GraphBasedFeatureExtractor):  # {{{
    def _extract(self, stack, deps, sent, i):
        return []
        fs = []
        if not stack:
            return fs
        w = stack[-1]
        built_childs = deps.all_childs
        possible_childs = set(self.childs[w['id']])
        for child in possible_childs - built_childs:
            #fs.append("s_cld_%s" % (self.toks[child]['tag']))
            #fs.append("s_cld_%s" % (self.toks[child]['form']))
            fs.append("st_s_cld_%s_%s" % (self.toks[child]['tag'], w['tag']))
            fs.append("st_s_cld_%s_%s" % (self.toks[child]['form'], w['tag']))
            #fs.append("sf_s_cld_%s_%s" % (self.toks[child]['tag'],w['form']))
            #fs.append("sf_s_cld_%s_%s" % (self.toks[child]['form'],w['form']))
        fs.append("s#pos_childs_%s" % len(possible_childs - built_childs))
        # fs.append("ts#pos_childs_%s_%s" % (w['tag'],len(possible_childs -
        # built_childs)))

        if not len(stack) > 1:
            return fs
        w = stack[-2]
        built_childs = deps.all_childs
        possible_childs = set(self.childs[w['id']])
        for child in possible_childs - built_childs:
            #fs.append("s1_cld_%s" % (self.toks[child]['tag']))
            #fs.append("s1_cld_%s" % (self.toks[child]['form']))
            fs.append("s1t_s_cld_%s_%s" % (self.toks[child]['tag'], w['tag']))
            fs.append("s1t_s_cld_%s_%s" % (self.toks[child]['form'], w['tag']))
            #fs.append("s1f_s_cld_%s_%s" % (self.toks[child]['tag'],w['form']))
            #fs.append("s1f_s_cld_%s_%s" % (self.toks[child]['form'],w['form']))
        fs.append("s1#pos_childs_%s" % len(possible_childs - built_childs))
        # fs.append("ts1#pos_childs_%s_%s" % ((w['tag']),len(possible_childs -
        # built_childs)))
        return fs
#}}}

# Working baselines #{{{


class WenbinFeatureExtractor:  # {{{
    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        features.append("s_%s" % s)
        features.append("s1_%s" % s1)
        features.append("w_%s" % w)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        features.append("Tss_%s_%s" % (Ts, s))
        features.append("Ts1s1_%s_%s" % (Ts1, s1))
        features.append("Tww_%s_%s" % (Tw, w))

        # bigram
        features.append("ss1_%s_%s" % (s, s1))  # @
        features.append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        features.append("sTss1_%s_%s_%s" % (s, Ts, s1))  # @
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        features.append("ss1Ts1_%s_%s_%s" % (s, s1, Ts1))  # @
        features.append("sTss1Ts1_%s_%s_%s_%s" % (s, Ts, s1, Ts1))  # @
        features.append("TsTw_%s_%s" % (Ts, Tw))
        features.append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))
        # more bigrams! [look at next word FORM]  # with these, 87.45 vs 87.09,
        # train 15-18, test 22
        features.append("ws1_%s_%s" % (w, s1))  # @
        features.append("ws_%s_%s" % (w, s))  # @
        features.append("wTs1_%s_%s" % (w, Ts1))  # @
        features.append("wTs_%s_%s" % (w, Ts))  # @

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts1sTw1_%s_%s_%s" % (Ts1, s, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        features.append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}


class WenbinFeatureExtractor_plus:  # {{{
    """
    like WenbinFeatureExtractor but include also POS of sent[i-1],sent[i-2]
    """

    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]
        wm1 = sent[i - 1] if i - 1 > 0 else PAD
        wm2 = sent[i - 2] if i - 2 > 0 else PAD

        Twm1 = wm1['tag']
        Twm2 = wm2['tag']

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        features.append("s_%s" % s)
        features.append("s1_%s" % s1)
        features.append("w_%s" % w)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        features.append("Tss_%s_%s" % (Ts, s))
        features.append("Ts1s1_%s_%s" % (Ts1, s1))
        features.append("Tww_%s_%s" % (Tw, w))

        #@NEW 4
        features.append("Twm1_%s" % Twm1)
        features.append("Twm2_%s" % Twm2)
        features.append("Twm1_%s_%s" % (Twm1, wm1['form']))
        features.append("Twm2_%s_%s" % (Twm2, wm1['form']))

        # bigram
        features.append("ss1_%s_%s" % (s, s1))  # @
        features.append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        features.append("sTss1_%s_%s_%s" % (s, Ts, s1))  # @
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        features.append("ss1Ts1_%s_%s_%s" % (s, s1, Ts1))  # @
        features.append("sTss1Ts1_%s_%s_%s_%s" % (s, Ts, s1, Ts1))  # @
        features.append("TsTw_%s_%s" % (Ts, Tw))
        features.append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))
        # more bigrams! [look at next word FORM]  # with these, 87.45 vs 87.09,
        # train 15-18, test 22
        features.append("ws1_%s_%s" % (w, s1))  # @
        features.append("ws_%s_%s" % (w, s))  # @
        features.append("wTs1_%s_%s" % (w, Ts1))  # @
        features.append("wTs_%s_%s" % (w, Ts))  # @

        features.append("TwTwm1_%s_%s" % (Tw, Twm1))  # @NEW
        features.append("Twm1Twm2_%s_%s" % (Twm1, Twm2))  # @NEW

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts1sTw1_%s_%s_%s" % (Ts1, s, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        features.append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}


class Degree2FeatureExtractor:  # {{{
    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        features.append("s_%s" % s)
        features.append("s1_%s" % s1)
        features.append("w_%s" % w)
        features.append("w1_%s" % w1)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)
        features.append("Tw1_%s" % Tw1)

        features.append("Tlcs_%s" % Tlcs)
        features.append("Tlcs1_%s" % Tlcs1)
        features.append("Trcs_%s" % Trcs)
        features.append("Trcs1_%s" % Trcs1)

        # @@ TODO: feature expand
        fs = ["%s_%s" % (f1, f2) for f1 in features for f2 in features]

        return fs
    #}}}


class EagerWenbinFeatureExtractor:  # {{{
    """
    my adaptation of WenbinFeatureExtractor to work for arc-eager
    (trivial -- just shift the focus from stack-1,stack-2 to stack-1,input[0])
    """

    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i + 1]
        w1 = sent[i + 2]
        s = sent[i]
        s1 = stack[-1]
        s2 = stack[-2]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        features.append("s_%s" % s)
        features.append("s1_%s" % s1)
        features.append("w_%s" % w)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        features.append("Tss_%s_%s" % (Ts, s))
        features.append("Ts1s1_%s_%s" % (Ts1, s1))
        features.append("Tww_%s_%s" % (Tw, w))

        # bigram
        features.append("ss1_%s_%s" % (s, s1))  # @
        features.append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        features.append("sTss1_%s_%s_%s" % (s, Ts, s1))  # @
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        features.append("ss1Ts1_%s_%s_%s" % (s, s1, Ts1))  # @
        features.append("sTss1Ts1_%s_%s_%s_%s" % (s, Ts, s1, Ts1))  # @
        features.append("TsTw_%s_%s" % (Ts, Tw))
        features.append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))
        # more bigrams! [look at next word FORM]  # with these, 87.45 vs 87.09,
        # train 15-18, test 22
        features.append("ws1_%s_%s" % (w, s1))  # @
        features.append("ws_%s_%s" % (w, s))  # @
        features.append("wTs1_%s_%s" % (w, Ts1))  # @
        features.append("wTs_%s_%s" % (w, Ts))  # @

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts1sTw1_%s_%s_%s" % (Ts1, s, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        features.append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}


class EagerDegree2FeatureExtractor:  # {{{
    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i + 1]
        w1 = sent[i + 2]
        s = sent[i]
        s1 = stack[-1]
        s2 = stack[-2]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        features.append("s_%s" % s)
        features.append("s1_%s" % s1)
        features.append("w_%s" % w)
        features.append("w1_%s" % w1)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)
        features.append("Tw1_%s" % Tw1)

        features.append("Tlcs_%s" % Tlcs)
        features.append("Tlcs1_%s" % Tlcs1)
        features.append("Trcs_%s" % Trcs)
        features.append("Trcs1_%s" % Trcs1)

        # @@ TODO: feature expand
        fs = ["%s_%s" % (f1, f2) for f1 in features for f2 in features]

        return fs
    #}}}


class EagerZhangFeatureExtractor:  # {{{
    """
    arc-eager features from "Tale of two parsers"
    http://www.aclweb.org/anthology/D/D08/D08-1059.pdf
    table 3
    """

    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        st0 = stack[-1]
        n0 = sent[i]
        n1 = sent[i + 1]
        n2 = sent[i + 2]

        # stack top
        STw = st0['form']
        STt = st0['tag']
        features.append("STwt_%s%s" % (STw, STt))
        features.append("STw_%s" % (STw))
        features.append("STt_%s" % (STt))
        # current word
        N0w = n0['form']
        N0t = n0['tag']
        features.append("N0wt_%s%s" % (N0w, N0t))
        features.append("N0t_%s" % N0t)
        features.append("N0w_%s" % N0w)
        # next word
        N1w = n1['form']
        N1t = n1['tag']
        N2t = n2['tag']
        features.append("N1wt_%s%s" % (N1w, N1t))
        features.append("N1t_%s" % N1t)
        features.append("N1w_%s" % N1w)
        # ST and N0
        features.append("STwtN0wt_%s_%s_%s_%s" % (STw, STt, N0w, N0t))
        features.append("STwtN0w_%s_%s_%s" % (STw, STt, N0w))
        features.append("STwN0wt_%s_%s_%s" % (STw, N0w, N0t))
        features.append("STwtN0t_%s_%s_%s" % (STw, STt, N0t))
        features.append("STtN0wt_%s_%s_%s" % (STt, N0w, N0t))
        features.append("STwN0w_%s_%s" % (STw, N0w))
        features.append("STtN0t_%s_%s" % (STt, N0t))
        # pos bigram
        features.append("N0tN1t_%s_%s" % (N0t, N1t))
        # pos trigram
        STPt = deps.parent(st0)
        if STPt:
            STPt = STPt['tag']
        STRCt = deps.right_child(st0)
        if STRCt:
            STRCt = STRCt['tag']
        STLCt = deps.left_child(st0)
        if STLCt:
            STLCt = STLCt['tag']
        N0LCt = deps.left_child(n0)
        if N0LCt:
            N0LCt = N0LCt['tag']
        features.append("N0tN1tN2t_%s_%s_%s" % (N0t, N1t, N2t))
        features.append("STtN0tN1t_%s_%s_%s" % (STt, N0t, N1t))
        features.append("STPtSTtN0t_%s_%s_%s" % (STPt, STt, N0t))
        features.append("STtSTLCtN0t_%s_%s_%s" % (STt, STLCt, N0t))
        features.append("STtSTRCtN0t_%s_%s_%s" % (STt, STRCt, N0t))
        features.append("STtN0tN0LCt_%s_%s_%s" % (STt, N0t, N0LCt))
        # N0 word
        features.append("N0wN1tN2t_%s_%s_%s" % (N0w, N1t, N2t))
        features.append("STtN0wN1t_%s_%s_%s" % (STt, N0w, N1t))
        features.append("STPtSTtN0w_%s_%s_%s" % (STPt, STt, N0w))
        features.append("STtSTLCtN0w_%s_%s_%s" % (STt, STLCt, N0w))
        features.append("STtSTRCtN0w_%s_%s_%s" % (STt, STRCt, N0w))
        features.append("STtN0wN0LCt_%s_%s_%s" % (STt, N0w, N0LCt))

        return features
    #}}}


class ExtendedEagerZhangFeatureExtractor:  # {{{
    """
    arc-eager features from "Tale of two parsers"
    http://www.aclweb.org/anthology/D/D08/D08-1059.pdf
    table 3

    extended with additional features,
    for the "cut trees" parsing
    """

    def __init__(self, level=1):
        self.level = level
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        st0 = stack[-1]
        n0 = sent[i]
        n1 = sent[i + 1]
        n2 = sent[i + 2]
        st0par = deps.parent(st0)
        # Extended participants
        nm1 = sent[i - 1] if i - 1 > 0 else PAD
        nm2 = sent[i - 2] if i - 2 > 0 else PAD
        nm3 = sent[i - 3] if i - 3 > 0 else PAD
        st0par2 = deps.parent(st0par) if st0par is not None else None
        st0par3 = deps.parent(st0par2) if st0par2 is not None else None

        # stack top
        STw = st0['form']
        STt = st0['tag']
        features.append("STwt_%s%s" % (STw, STt))
        features.append("STw_%s" % (STw))
        features.append("STt_%s" % (STt))
        # current word
        N0w = n0['form']
        N0t = n0['tag']
        features.append("N0wt_%s%s" % (N0w, N0t))
        features.append("N0t_%s" % N0t)
        features.append("N0w_%s" % N0w)
        # next word
        N1w = n1['form']
        N1t = n1['tag']
        N2t = n2['tag']
        features.append("N1wt_%s%s" % (N1w, N1t))
        features.append("N1t_%s" % N1t)
        features.append("N1w_%s" % N1w)
        # ST and N0
        features.append("STwtN0wt_%s_%s_%s_%s" % (STw, STt, N0w, N0t))
        features.append("STwtN0w_%s_%s_%s" % (STw, STt, N0w))
        features.append("STwN0wt_%s_%s_%s" % (STw, N0w, N0t))
        features.append("STwtN0t_%s_%s_%s" % (STw, STt, N0t))
        features.append("STtN0wt_%s_%s_%s" % (STt, N0w, N0t))
        features.append("STwN0w_%s_%s" % (STw, N0w))
        features.append("STtN0t_%s_%s" % (STt, N0t))
        # pos bigram
        features.append("N0tN1t_%s_%s" % (N0t, N1t))
        # pos trigram
        STPt = st0par
        if STPt:
            STPt = STPt['tag']
        STRCt = deps.right_child(st0)
        if STRCt:
            STRCt = STRCt['tag']
        STLCt = deps.left_child(st0)
        if STLCt:
            STLCt = STLCt['tag']
        N0LCt = deps.left_child(n0)
        if N0LCt:
            N0LCt = N0LCt['tag']
        features.append("N0tN1tN2t_%s_%s_%s" % (N0t, N1t, N2t))
        features.append("STtN0tN1t_%s_%s_%s" % (STt, N0t, N1t))
        features.append("STPtSTtN0t_%s_%s_%s" % (STPt, STt, N0t))
        features.append("STtSTLCtN0t_%s_%s_%s" % (STt, STLCt, N0t))
        features.append("STtSTRCtN0t_%s_%s_%s" % (STt, STRCt, N0t))
        features.append("STtN0tN0LCt_%s_%s_%s" % (STt, N0t, N0LCt))
        # N0 word
        features.append("N0wN1tN2t_%s_%s_%s" % (N0w, N1t, N2t))
        features.append("STtN0wN1t_%s_%s_%s" % (STt, N0w, N1t))
        features.append("STPtSTtN0w_%s_%s_%s" % (STPt, STt, N0w))
        features.append("STtSTLCtN0w_%s_%s_%s" % (STt, STLCt, N0w))
        features.append("STtSTRCtN0w_%s_%s_%s" % (STt, STRCt, N0w))
        features.append("STtN0wN0LCt_%s_%s_%s" % (STt, N0w, N0LCt))

        # Extended
        Nm1t = nm1['tag']
        Nm2t = nm2['tag']
        Nm3t = nm3['tag']
        Nm1w = nm1['form']
        features.append("N-1tN0t_%s_%s" % (Nm1t, N0t))
        features.append("N-2tN-1t_%s_%s" % (Nm2t, Nm1t))
        features.append("N-3tN-2t_%s_%s" % (Nm3t, Nm2t))
        features.append("N-1tN0tN0w_%s_%s_%s" % (Nm1t, N0t, N0w))
        features.append("N-1tN-1wN0w_%s_%s_%s" % (Nm1t, Nm1w, N0t))
        # extended plus
        if self.level > 1:
            pars = []
            par = deps.parent(st0)
            while par is not None:
                pars.append(par)
                par = deps.parent(par)
            for par in pars:
                features.append("stppt_N0t_%s_%s" % (par['tag'], N0t))
                features.append(
                    "stppt_N0tN0w_%s_%s_%s" %
                    (par['tag'], N0t, N0w))
                features.append(
                    "stpptw_N0t_%s_%s_%s" %
                    (par['tag'], par['form'], N0t))
                features.append(
                    "stpptw_N0tN0w_%s_%s_%s_%s" %
                    (par['tag'], par['form'], N0t, N0w))
        # extended plusplus
        if self.level > 2:
            _top = pars[-1] if pars else st0
            if _top is not NOPARENT:
                _idx = stack.index(_top)
                prev = stack[_idx - 1] if _idx - 1 > 0 else PAD
            else:
                prev = NOPARENT
            features.append("stPRVt_st0t_%s_%s" % (prev['tag'], STt))
            features.append("stPRVt_stTopt_%s_%s" % (prev['tag'], _top['tag']))
            features.append("stPRVt_stTopt_%s_%s" % (_top['tag'], STt))
        # extended plus-plus-plus: looking at all valid previous tokens
        if self.level > 3:
            lencurrent = 0
            for _tok in sent[i - 1:0:-1]:
                if deps.parent(_tok) is None:
                    lencurrent = sent[i]['id'] - _tok['id']
                    break
            for x in (0, 1, 2, 3, 5, 7, 10):
                if lencurrent > x:
                    features.append("len>%s" % x)
                else:
                    features.append("len<=%s" % x)

            # sent[i]['__par']=-99
            # for t in sent[:i]:
            #   par = t['__par']
            #   if par==-99:
            #      par = deps.parent(t)
            #      t['__par']=par
            #   if (par is None) or par['id'] < t['id']:
            #      features.append("PRVt_st0t_%s_%s" % (t['tag'],STt))
            #      features.append("PRVt_n0t_%s_%s" % (t['tag'],N0t))
            #      features.append("PRVt_st0t_%s_%s_%s_%s" % (t['tag'],STt,t['form'],STw))
            #      features.append("PRVt_n0t_%s_%s_%s_%s" % (t['tag'],N0t,t['form'],N0w))

        return features
    #}}}


class EagerMaltFeatureExtractor:  # {{{
    """
    Arabic malt-parser features
       Based on ara.par, without the morph/lem/dep features
    """

    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        """ {{{
      POS	STACK
      POS	INPUT
      POS	INPUT	1
      POS	INPUT	2
      POS	STACK	1
      POS	STACK	0  	0	   0	   -1	   1
      POS	INPUT	0	   -1
      CPOS	STACK
      CPOS	INPUT
      CPOS	STACK	0	   0	   1	   -1
      DEP	STACK
      FEATS	STACK
      FEATS	INPUT
      LEMMA	STACK
      LEMMA	INPUT
      LEX	STACK
      LEX	INPUT
      LEX	INPUT	1
      LEX	STACK	0	   0	   1
      LEX	INPUT	0	   -1
      LEX	STACK	0	   0	   0	   1	   -1
      """  # }}}

        # participants
        in0 = sent[i]
        in1 = sent[i + 1]
        in2 = sent[i + 2]
        st0 = stack[-1]
        st1 = stack[-2]
        st000_11 = deps.sibling(deps.left_child(st0), 1)
        in0_1 = sent[i - 1]
        st001 = deps.parent(st0)
        st001_1 = deps.left_child(st001)
        st0001_1 = deps.sibling(deps.right_child(st0), -1)
        if not st001:
            st001 = {'tag': None, 'ctag': None, 'form': None}
        if not st000_11:
            st000_11 = {'tag': None, 'ctag': None, 'form': None}
        if not st001_1:
            st001_1 = {'tag': None, 'ctag': None, 'form': None}
        if not st0001_1:
            st0001_1 = {'tag': None, 'ctag': None, 'form': None}

        features.append("Tst0_%s" % st0['tag'])
        features.append("Tin0_%s" % in0['tag'])
        features.append("Tin1_%s" % in1['tag'])
        features.append("Tin2_%s" % in2['tag'])
        features.append("Tst1_%s" % st1['tag'])
        features.append("Tst000-11_%s" % st000_11['tag'])
        features.append("Tin0-1_%s" % in0_1['tag'])
        features.append("CTst0_%s" % st0['ctag'])
        features.append("CTin0_%s" % in0['ctag'])
        features.append("CTst001_1_%s" % st001_1['ctag'])
        # dep_st0 -- skipped
        # feats, lemmas: skipped
        features.append("Lst0_%s" % st0['form'])
        features.append("Lin0_%s" % in0['form'])
        features.append("Lin1_%s" % in1['form'])
        features.append("Lst001_%s" % st001['form'])
        features.append("Lin0-1_%s" % in0_1['form'])
        features.append("Lst0001-1_%s" % st0001_1['form'])
        if 'morph' in st0:
            for f in st0['morph']:
                features.append("Mst0_%s" % f)
        if 'morph' in in0:
            for f in in0['morph']:
                features.append("Min0_%s" % f)

        if 'lem' in in0:
            features.append("LMin0_%s" % in0['lem'])
        if 'lem' in st0:
            features.append("LMst0_%s" % st0['lem'])

        # @@ TODO: feature expand
        fs = ["%s_%s" % (f1, f2) for f1 in features for f2 in features]

        return fs
    #}}}


class EagerMaltFeatureExtractor:  # {{{
    """
    Arabic malt-parser features
       Based on ara.par, without the morph/lem/dep features
    """

    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 3:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        """ {{{
      POS	STACK
      POS	INPUT
      POS	INPUT	1
      POS	INPUT	2
      POS	STACK	1
      POS	STACK	0  	0	   0	   -1	   1
      POS	INPUT	0	   -1
      CPOS	STACK
      CPOS	INPUT
      CPOS	STACK	0	   0	   1	   -1
      DEP	STACK
      FEATS	STACK
      FEATS	INPUT
      LEMMA	STACK
      LEMMA	INPUT
      LEX	STACK
      LEX	INPUT
      LEX	INPUT	1
      LEX	STACK	0	   0	   1
      LEX	INPUT	0	   -1
      LEX	STACK	0	   0	   0	   1	   -1
      """  # }}}

        # participants
        in0 = sent[i]
        in1 = sent[i + 1]
        in2 = sent[i + 2]
        st0 = stack[-1]
        st1 = stack[-2]
        st000_11 = deps.sibling(deps.left_child(st0), 1)
        in0_1 = sent[i - 1]
        st001 = deps.parent(st0)
        st001_1 = deps.left_child(st001)
        st0001_1 = deps.sibling(deps.right_child(st0), -1)
        if not st001:
            st001 = {'tag': None, 'ctag': None, 'form': None}
        if not st000_11:
            st000_11 = {'tag': None, 'ctag': None, 'form': None}
        if not st001_1:
            st001_1 = {'tag': None, 'ctag': None, 'form': None}
        if not st0001_1:
            st0001_1 = {'tag': None, 'ctag': None, 'form': None}

        features.append("Tst0_%s" % st0['tag'])
        features.append("Tin0_%s" % in0['tag'])
        features.append("Tin1_%s" % in1['tag'])
        features.append("Tin2_%s" % in2['tag'])
        features.append("Tst1_%s" % st1['tag'])
        features.append("Tst000-11_%s" % st000_11['tag'])
        features.append("Tin0-1_%s" % in0_1['tag'])
        features.append("CTst0_%s" % st0['ctag'])
        features.append("CTin0_%s" % in0['ctag'])
        features.append("CTst001_1_%s" % st001_1['ctag'])
        # dep_st0 -- skipped
        # feats, lemmas: skipped
        features.append("Lst0_%s" % st0['form'])
        features.append("Lin0_%s" % in0['form'])
        features.append("Lin1_%s" % in1['form'])
        features.append("Lst001_%s" % st001['form'])
        features.append("Lin0-1_%s" % in0_1['form'])
        features.append("Lst0001-1_%s" % st0001_1['form'])
        if 'morph' in st0:
            for f in st0['morph']:
                features.append("Mst0_%s" % f)
        if 'morph' in in0:
            for f in in0['morph']:
                features.append("Min0_%s" % f)

        if 'lem' in in0:
            features.append("LMin0_%s" % in0['lem'])
        if 'lem' in st0:
            features.append("LMst0_%s" % st0['lem'])

        # @@ TODO: feature expand
        fs = ["%s_%s" % (f1, f2) for f1 in features for f2 in features]

        return fs
    #}}}


class EagerMaltEnglishFeatureExtractor:  # {{{
    """
    English malt-parser features
       Based on eng.par, without the dep features
    """

    def __init__(self, allpairs=False, words=None):
        self.allpairs = allpairs
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 4:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        """ {{{
POS	STACK
POS	INPUT
POS	INPUT	1
POS	INPUT	2
POS	INPUT	3
POS	STACK	1
POS	STACK	0	0	0	-1
POS	INPUT	0	0	0	-1
CPOS	STACK
CPOS	INPUT
CPOS	STACK	0	-1
DEP	STACK
DEP	STACK	0	0	0	-1
DEP	STACK	0	0	0	1
DEP	INPUT	0	0	0	-1
LEX	STACK
LEX	INPUT
LEX	INPUT	1
LEX	STACK	0	0	1
      """  # }}}

        # participants
        in0 = sent[i]
        in1 = sent[i + 1]
        in2 = sent[i + 2]
        in3 = sent[i + 3]
        st0 = stack[-1]
        st1 = stack[-2]

        st000_1 = deps.left_child(st0)  # left child of stack
        in000_1 = deps.left_child(in0)
        if st0['id'] == 0 or st0 == PAD:
            st0_1 = PAD
        else:
            # token just before top-of-stack in input string
            st0_1 = sent[st0['id'] - 1]
            assert(st0_1['id'] == st0['id'] -
                   1), "%s %s" % (st0_1['id'], st0['id'] - 1)
        st0001 = deps.right_child(st0)  # right child of stack
        st001 = deps.parent(st0)

        if not st001:
            st001 = {'tag': None, 'ctag': None, 'form': None}
        if not st000_1:
            st000_1 = {'tag': None, 'ctag': None, 'form': None}
        if not in000_1:
            in000_1 = {'tag': None, 'ctag': None, 'form': None}
        if not st0_1:
            st0_1 = {'tag': None, 'ctag': None, 'form': None}
        if not st0001:
            st0001 = {'tag': None, 'ctag': None, 'form': None}

        f = features.append

        f("ps_%s" % st0['tag'])
        f("pi_%s" % in0['tag'])
        f("pi1_%s" % in1['tag'])
        f("pi2_%s" % in2['tag'])
        f("pi3_%s" % in3['tag'])
        f("ps1_%s" % st1['tag'])
        f("ps000-1_%s" % st000_1['tag'])
        f("pi000-1_%s" % in000_1['tag'])
        f("cps_%s" % st0['ctag'])
        f("cpi_%s" % in0['ctag'])
        f("cps0-1_%s" % st0_1['ctag'])
        # dep_st... -- skipped
        f("ls_%s" % st0['form'])
        f("li_%s" % in0['form'])
        f("li1_%s" % in1['form'])
        f("ls001_%s" % st001['form'])

        # @@ TODO: feature expand
        if self.allpairs:
            fs = ["%s_%s" % (f1, f2) for f1 in features for f2 in features]
            return fs
        return features
    #}}}


class WenbinFeatureExtractor2:  # {{{
    def __init__(self):

        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))
        append = features.append

        # participants
        w = sent[i] if len(sent) > i else PAD
        w1 = sent[i + 1] if len(sent) > i + 1 else PAD
        s = stack[-1] if len(stack) > 0 else PAD
        s1 = stack[-2] if len(stack) > 1 else PAD
        s2 = stack[-3] if len(stack) > 2 else PAD

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram
        append("s_%s" % s)
        append("s1_%s" % s1)
        append("w_%s" % w)

        append("Ts_%s" % Ts)
        append("Ts1_%s" % Ts1)
        append("Tw_%s" % Tw)

        append("Tss_%s_%s" % (Ts, s))
        append("Ts1s1_%s_%s" % (Ts1, s1))
        append("Tww_%s_%s" % (Tw, w))

        # bigram
        append("ss1_%s_%s" % (s, s1))
        append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        append("sTss1_%s_%s_%s" % (s, Ts, s1))
        append("TsTs1_%s_%s" % (Ts, Ts1))
        append("ss1Ts1_%s_%s_%s" % (s, s1, Ts1))
        append("sTss1Ts1_%s_%s_%s_%s" % (s, Ts, s1, Ts1))
        append("TsTw_%s_%s" % (Ts, Tw))
        append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))

        # trigram
        append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        append("Ts1sTw1_%s_%s_%s" % (Ts1, s, Tw))
        append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}

#}}}


class UnlexFeatureExtractor:  # {{{ 83.81 train 15-18 test 22
    def __init__(self):
        self.last_sent = None

    def extract(self, stack, deps, sent, i):
        features = []

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']

        Tw1 = w1['tag']

        Ts = s['tag']

        Ts1 = s1['tag']

        Ts2 = s2['tag']

        # unigram
        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        # bigram
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        features.append("TsTw_%s_%s" % (Ts, Tw))

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))

        return features
    #}}}


from collections import defaultdict


class UnlexWenbinPlusFeatureExtractor:  # {{{ Good one!
    def __init__(self):
        self.dat = {}
        datafile = open(os.path.join(os.path.dirname(__file__), "data"), "r")
        for line in datafile.readlines():
            line = line.strip().split()
            self.dat[(line[0], line[1])] = int(line[2])


        self.dat = defaultdict(int, [(k, v)
                                     for k, v in self.dat.items() if v > 5])

    def extract(self, stack, deps, sent, i):
        features = []
        import math
        # new features, which I think helps..
        #features.append("toend_%s" % round(math.log(len(sent)+3-i)))

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

#      if Ts1=='IN':
#         features.append("1par_s1_s:%s" % (self.dat[(s1,s)],))
#         features.append("1par_s_s1:%s" % (self.dat[(s,s1)],))
#         features.append("1par_s1_w:%s" % (self.dat[(s1,w)],))
#         features.append("1par_w_s1:%s" % (self.dat[(w,s1)],))
#      if Ts == 'IN':
#         features.append("2par_s1_s:%s" % (self.dat[(s1,s)],))
#         features.append("2par_s_s1:%s" % (self.dat[(s,s1)],))
#         features.append("2par_s_w:%s" % (self.dat[(s,w)],))
#         features.append("2par_w_s:%s" % (self.dat[(w,s)],))
#      if Tw=='IN':
#         features.append("3par_w_s1:%s" % (self.dat[(w,s1)],))
#         features.append("3par_s1_w:%s" % (self.dat[(s1,w)],))
#         features.append("3par_w_s:%s" % (self.dat[(w,s)],))
#         features.append("3par_s_w:%s" % (self.dat[(s,w)],))
        if Tw == 'IN':
            if (s1, w) in self.dat or (s, w) in self.dat:
                features.append("m1_%s" %
                                (self.dat[(s1, w)] > self.dat[(s, w)]))
            else:
                features.append("m1_NA")
        # if Ts=='IN':
        #   features.append("m2_%s" % (self.dat[(s1,s)]-self.dat[(w,s)]))
        # if Ts1=='IN':
        #   features.append("m3_%s" % (self.dat[(s,s1)]-self.dat[(w,s1)]))

        # unigram 87,71 (conditioning) vs 87,5 (removing)
        if Ts[0] not in 'JNV':
            features.append("s_%s" % s)
        if Ts1[0] not in 'JNV':
            features.append("s1_%s" % s1)
        if Tw[0] not in 'JNV':
            features.append("w_%s" % w)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        #features.append("Tss_%s_%s" % (Ts,s))
        #features.append("Ts1s1_%s_%s" % (Ts1,s1))
        #features.append("Tww_%s_%s" % (Tw,w))

        # bigram
        # features.append("ss1_%s_%s" % (s,s1))                    #@
        features.append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        # features.append("sTss1_%s_%s_%s" % (s,Ts,s1))            #@
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        # features.append("ss1Ts1_%s_%s_%s" % (s,s1,Ts1))          #@
        # features.append("sTss1Ts1_%s_%s_%s_%s" % (s,Ts,s1,Ts1))  #@
        features.append("TsTw_%s_%s" % (Ts, Tw))
        features.append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))
        # more bigrams! [look at next word FORM]  # with these, 87.45 vs 87.09, train 15-18, test 22
        # features.append("ws1_%s_%s" % (w,s1))                    #@
        # features.append("ws_%s_%s" % (w,s))                    #@
        features.append("wTs1_%s_%s" % (w, Ts1))  # @
        features.append("wTs_%s_%s" % (w, Ts))  # @

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts1sTw_%s_%s_%s" % (Ts1, s, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        features.append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}


class BestSoFarFeatureExtractor:  # {{{ 87.71 train 15-18 test 22
    def __init__(self):
        pass

    def extract(self, stack, deps, sent, i):
        features = []
        import math

        if len(sent) < i + 2:
            sent = sent[:]
            sent.append(PAD)
            sent.append(PAD)
        if len(stack) < 3:
            stack = [PAD, PAD, PAD] + stack

        # participants
        w = sent[i]
        w1 = sent[i + 1]
        s = stack[-1]
        s1 = stack[-2]
        s2 = stack[-3]

        Tlcs1 = deps.left_child(s1)
        if Tlcs1:
            Tlcs1 = Tlcs1['tag']

        Tlcs = deps.left_child(s)
        if Tlcs:
            Tlcs = Tlcs['tag']

        Trcs = deps.right_child(s)
        if Trcs:
            Trcs = Trcs['tag']

        Trcs1 = deps.right_child(s1)
        if Trcs1:
            Trcs1 = Trcs1['tag']

        Tw = w['tag']
        w = w['form']

        Tw1 = w1['tag']
        w1 = w1['form']

        Ts = s['tag']
        s = s['form']

        Ts1 = s1['tag']
        s1 = s1['form']

        Ts2 = s2['tag']
        s2 = s2['form']

        # unigram 87,71 (conditioning) vs 87,5 (removing)
        if Ts[0] not in 'JNV':
            features.append("s_%s" % s)
        if Ts1[0] not in 'JNV':
            features.append("s1_%s" % s1)
        if Tw[0] not in 'JNV':
            features.append("w_%s" % w)

        features.append("Ts_%s" % Ts)
        features.append("Ts1_%s" % Ts1)
        features.append("Tw_%s" % Tw)

        #features.append("Tss_%s_%s" % (Ts,s))
        #features.append("Ts1s1_%s_%s" % (Ts1,s1))
        #features.append("Tww_%s_%s" % (Tw,w))

        # bigram
        # features.append("ss1_%s_%s" % (s,s1))                    #@
        features.append("Tss1Ts1_%s_%s_%s" % (Ts, s1, Ts1))
        # features.append("sTss1_%s_%s_%s" % (s,Ts,s1))            #@
        features.append("TsTs1_%s_%s" % (Ts, Ts1))
        # features.append("ss1Ts1_%s_%s_%s" % (s,s1,Ts1))          #@
        # features.append("sTss1Ts1_%s_%s_%s_%s" % (s,Ts,s1,Ts1))  #@
        features.append("TsTw_%s_%s" % (Ts, Tw))
        features.append("sTsTs1_%s_%s_%s" % (s, Ts, Ts1))
        # more bigrams! [look at next word FORM]  # with these, 87.45 vs 87.09, train 15-18, test 22
        # features.append("ws1_%s_%s" % (w,s1))                    #@
        # features.append("ws_%s_%s" % (w,s))                    #@
        features.append("wTs1_%s_%s" % (w, Ts1))  # @
        features.append("wTs_%s_%s" % (w, Ts))  # @

        # trigram
        features.append("TsTwTw1_%s_%s_%s" % (Ts, Tw, Tw1))
        features.append("sTwTw1_%s_%s_%s" % (s, Tw, Tw1))
        features.append("Ts1TsTw_%s_%s_%s" % (Ts1, Ts, Tw))
        features.append("Ts1sTw_%s_%s_%s" % (Ts1, s, Tw))
        features.append("Ts2Ts1Ts_%s_%s_%s" % (Ts2, Ts1, Ts))

        # modifier
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTrcs_%s_%s_%s" % (Ts1, Ts, Trcs))
        features.append("Ts1sTlcs_%s_%s_%s" % (Ts1, s, Tlcs))
        features.append("Ts1Trcs1Ts_%s_%s_%s" % (Ts1, Trcs1, Ts))
        features.append("Ts1Tlcs1Ts_%s_%s_%s" % (Ts1, Tlcs1, Ts))
        features.append("Ts1TsTlcs_%s_%s_%s" % (Ts1, Ts, Tlcs))
        features.append("Ts1Trcs1s_%s_%s_%s" % (Ts1, Trcs1, s))

        return features
    #}}}

#}}}


__EXTRACTORS__['eager.zhang'] = EagerZhangFeatureExtractor()
__EXTRACTORS__['eager.zhang.ext'] = ExtendedEagerZhangFeatureExtractor()
__EXTRACTORS__['eager.zhang.ext2'] = ExtendedEagerZhangFeatureExtractor(2)
__EXTRACTORS__['eager.zhang.ext3'] = ExtendedEagerZhangFeatureExtractor(3)
__EXTRACTORS__['eager.zhang.ext4'] = ExtendedEagerZhangFeatureExtractor(4)
__EXTRACTORS__['eager.malt.eng'] = EagerMaltEnglishFeatureExtractor(
    allpairs=True)

__EXTRACTORS__['standard.wenbin'] = WenbinFeatureExtractor()  # Good one
__EXTRACTORS__['standard.wenbinplus'] = WenbinFeatureExtractor_plus()  # Good one
__EXTRACTORS__['standard.deg2'] = Degree2FeatureExtractor()
__EXTRACTORS__['standard.unlex.wb'] = UnlexWenbinPlusFeatureExtractor()
__EXTRACTORS__['standard.unlex'] = UnlexFeatureExtractor()


def get(name):
    try:
        return __EXTRACTORS__[name]
    except KeyError:
        import sys
        sys.stderr.write(
            "invalid feature extactor %s. possible values: %s\n" %
            (name, __EXTRACTORS__.keys()))
        sys.exit()
