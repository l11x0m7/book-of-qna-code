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
Transition based parsing (both arc-standard and arc-eager).
Easily extended to support other variants.
"""
from __future__ import print_function
from __future__ import division

import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(curdir, os.path.pardir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    # raise "Must be using Python 3"

from absl import app
from absl import flags
from absl import logging
from collections import defaultdict
import copy
import random
from ml import ml

from common import *
from common.exceptions import *
from configurations import *

# misc / pretty print / temp junk

def _ids(tok):
    if tok['id'] == 0:
        tok['form'] = 'ROOT'
    return (tok['id'], tok['tag'])



class MLActionDecider:
    '''
    action deciders / policeis
    '''

    def __init__(self, model, featExt):
        self.m = model
        self.fs = featExt

    def next_action(self, stack, deps, sent, i):
        if len(stack) < 2:
            return SHIFT
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        if i >= len(sent) and action == SHIFT:
            action = scores.index(max(scores[1:]))
        return action

    def next_actions(self, stack, deps, sent, i, conf=None):
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        # [-122, 0.3, 3] -> {0:-122, 1:0.3, 2:3}
        scores = dict(enumerate(scores))
        actions = [
            item for item,
            score in sorted(
                scores.items(),
                key=lambda x:-
                x[1])]
        return actions

    def scores(self, conf):  # TODO: who uses this??
        if len(conf.stack) < 2:
            return {SHIFT: 1}

        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        action, scores = self.m.predict(fs)
        # [-122, 0.3, 3] -> {0:-122, 1:0.3, 2:3}
        scores = dict(enumerate(scores))
        if conf.i >= len(conf.sent):
            del scores[SHIFT]
        return scores

    def get_scores(self, conf):
        if len(conf.stack) < 2:
            return {SHIFT: 1}
        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        scores = self.m.get_scores(fs)
        return scores

    def get_prob_scores(self, conf):
        if len(conf.stack) < 2:
            return [1.0, 0, 0]
        fs = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        besti, scores = self.m.predict(fs)
        return scores

    def update(self, stack, deps, sent, i):
        self.m.update(wrong, correct, self.fs.extract(stack, deps, sent, i))


class OracleActionDecider:  # {{{
    def __init__(self, oracle):
        self.o = oracle

    def next_action(self, stack, deps, sent, i):
        return self.o.next_action(stack, deps, sent, i)

    def next_actions(self, stack, deps, sent, i):
        return self.o.next_actions(stack, deps, sent, i)

    def get_scores(self, conf):
        return {self.next_action(conf.stack, conf.deps, conf.sent, conf.i): 1}


class AuditMLActionDecider:  # {{{
    def __init__(self, model, featExt):
        self.m = model
        self.fs = featExt

        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        self.idtotok = {}

    def next_action(self, stack, deps, sent, i):
        def _enrich(set_of_node_ids):
            return [_ids(self.idtotok[i]) for i in set_of_node_ids]

        if self.current_sent != sent:
            self.current_sent = sent
            idtotok = {}
            for tok in self.current_sent:
                self.idtotok[tok['id']] = tok
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        if len(stack) < 2:
            return SHIFT
        fs = self.fs.extract(stack, deps, sent, i)
        action, scores = self.m.predict(fs)
        logging.debug("action [%s], scores [%s]", action, scores)
        if i >= len(sent) and action == SHIFT:
            action = scores.index(max(scores[1:]))

        if action == REDUCE_R:
            if stack[-1]['parent'] == stack[-2]['id']:
                if len(self.childs_of_x[stack[-1]['id']
                                        ] - self.connected_childs) > 0:
                    logging.error("R not connecting: %s | %s , because: %s", _ids(stack[-1]), _ids(stack[-2]), _enrich(self.childs_of_x[stack[-1]['id']] - self.connected_childs))
                else:
                    logging.error("R not XX")

        if action == REDUCE_L:
            if len(self.childs_of_x[stack[-2]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-2]['id'])
        if action == REDUCE_R:
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-1]['id'])

        return action, scores

class LoggingActionDecider:  # {{{
    def __init__(self, decider, featExt, out=sys.stdout):
        self.decider = decider
        self.fs = featExt
        self.out = out

    def next_action(self, stack, deps, sent, i):
        features = self.fs.extract(stack, deps, sent, i)
        logging.debug("features [%s]", features) 
        action = self.decider.next_action(stack, deps, sent, i)
        self.out.write("%s %s\n" % (action, " ".join(features)))
        return action

    def next_actions(self, stack, deps, sent, i):
        action = self.next_action(stack, deps, sent, i)
        return [action]

    def save(self, param=None):
        self.out.close()


class MLTrainerActionDecider:  # {{{
    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def next_action(self, stack, deps, sent, i, conf=None):
        action = self.decider.next_action(stack, deps, sent, i)
        mlaction = self.ml.update(
            action, self.fs.extract(
                stack, deps, sent, i))
        if action != mlaction:
            if self.earlyUpdate:
                raise MLTrainerWrongActionException()
        return action

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLPassiveAggressiveTrainerActionDecider:  # {{{
    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i):
        return [self.next_action(stack, deps, sent, i)]

    def next_action(self, stack, deps, sent, i):
        action = self.decider.next_action(stack, deps, sent, i)
        mlaction = self.ml.do_pa_update(
            self.fs.extract(
                stack, deps, sent, i), action, C=1.0)
        if action != mlaction:
            if self.earlyUpdate:
                raise MLTrainerWrongActionException()
        return action

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLTrainerActionDecider2:  # {{{
    """
    Like MLTrainerActionDecider but does the update itself (a little less efficient, a bit more informative)
    """

    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def score_deps(self, deps, sent):
        score = 0
        deps = deps.deps  # a set of (p,c) ids
        sent_deps = set()
        for tok in sent:
            if tok['id'] == 0:
                continue
            sent_deps.add((tok['parent'], tok['id']))
        for pc in sent_deps:
            if pc not in deps:
                score += 0.2
        for pc in deps:
            if pc not in sent_deps:
                score += 1
        return score

    def cum_score_of_action(self, action, conf, ml=False):
        newconf = conf.newAfter(action)
        decider = copy.deepcopy(self.decider)
        while not newconf.is_in_finish_state():
            try:
                if ml:
                    next = self.next_ml_action(newconf)
                else:
                    next = decider.next_action(
                        newconf.stack, newconf.deps, newconf.sent, newconf.i)
                newconf.do_action(next)
            except IllegalActionException:
                assert(len(newconf.sent) == newconf.i)
                break
        return self.score_deps(newconf.deps, newconf.sent)

    def next_ml_action(self, conf):
        features = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        return max(act_scores)[1]

    def next_action(self, stack, deps, sent, i, conf=None):
        features = self.fs.extract(stack, deps, sent, i)
        goldaction = self.decider.next_action(stack, deps, sent, i)

        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        pred_s, pred_a = max(act_scores)
        self.ml.tick()
        if pred_a != goldaction:
            # calculate cost of NO UPDATE:
            noupdate_cost = self.cum_score_of_action(pred_a, conf, ml=True)

            # now try to update:
            self.ml.add(features, goldaction, 1.0)
            self.ml.add(features, pred_a, -1.0)

            update_cost = self.cum_score_of_action(
                self.next_ml_action(conf), conf, ml=True)
            if noupdate_cost < update_cost:
                logging.debug("noupdate: %s, update: %s", noupdate_cost, update_cost)
                # undo prev update
                self.ml.add(features, goldaction, -1.0)
                self.ml.add(features, pred_a, 1.0)
        return goldaction

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)

class MLTrainerActionDecider3:  # {{{
    """
    Like MLTrainerActionDecider but does the update itself (a little less efficient, a bit more informative)
    """

    def __init__(self, mlAlgo, decider, featExt, earlyUpdate=False):
        self.decider = decider
        self.ml = mlAlgo
        self.fs = featExt
        self.earlyUpdate = earlyUpdate

    def next_actions(self, stack, deps, sent, i, conf=None):
        return [self.next_action(stack, deps, sent, i, conf)]

    def score_deps(self, deps, sent):
        score = 0
        deps = deps.deps  # a set of (p,c) ids
        sent_deps = set()
        for tok in sent:
            if tok['id'] == 0:
                continue
            sent_deps.add((tok['parent'], tok['id']))
        for pc in sent_deps:
            if pc not in deps:
                score += 0.2
        for pc in deps:
            if pc not in sent_deps:
                score += 1
        return score

    def cum_score_of_action(self, action, conf, ml=False):
        newconf = conf.newAfter(action)
        decider = copy.deepcopy(self.decider)
        while not newconf.is_in_finish_state():
            try:
                if ml:
                    next = self.next_ml_action(newconf)
                else:
                    next = decider.next_action(
                        newconf.stack, newconf.deps, newconf.sent, newconf.i)
                newconf.do_action(next)
            except IllegalActionException:
                logging.debug("oracle says [%s], but it is illegal, probably at end", next) 
                assert(len(newconf.sent) == newconf.i)
                break
        return self.score_deps(newconf.deps, newconf.sent)

    def next_ml_action(self, conf):
        features = self.fs.extract(conf.stack, conf.deps, conf.sent, conf.i)
        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]
        return max(act_scores)[1]

    def next_action(self, stack, deps, sent, i, conf=None):
        features = self.fs.extract(stack, deps, sent, i)
        goldaction = self.decider.next_action(stack, deps, sent, i)

        act_scores = [(score, act) for act, score in self.ml.get_scores(
            features).items() if act in conf.valid_actions()]

        pred_s, pred_a = max(act_scores)
        noupdate_cost = self.cum_score_of_action(pred_a, conf, ml=True)

        if pred_a != SHIFT:
            self.ml.add(features, SHIFT, 1.0)
            self.ml.add(features, pred_a, -1.0)
            shiftupdate_cost = self.cum_score_of_action(SHIFT, conf, ml=True)
            if shiftupdate_cost < noupdate_cost:
                self.ml.tick()
                return SHIFT
            else:  # undo
                self.ml.add(features, SHIFT, -1.0)
                self.ml.add(features, pred_a, 1.0)
                self.ml.tick()
                return pred_a
        self.ml.tick()
        return pred_a

        costs = []
        for score, act in act_scores:
            if pred_a != act:
                self.ml.add(features, act, 1.0)
                self.ml.add(features, pred_a, -1.0)
                costs.append(
                    (self.cum_score_of_action(
                        act, conf, ml=True), act))
                self.ml.add(features, act, -1.0)
                self.ml.add(features, pred_a, 1.0)
            else:
                costs.append((noupdate_cost, pred_a))
        min_cost, act = min(costs)

        if act != pred_a:
            logging.debug("min_cost [%s], noupdate_cost [%s], act [%s], goldaction [%s]", min_cost, noupdate_cost, act, goldaction)
            self.ml.add(features, act, 1.0)
            self.ml.add(features, pred_a, -1.0)
        else:
            pass

        self.ml.tick()
        return act

    def save(self, fout):
        self.ml.finalize()
        self.ml.dump(fout)
