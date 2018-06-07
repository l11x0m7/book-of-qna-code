#!/usr/bin/env python
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
driver for ArcEager parser.

Author: Yoav Goldberg (yoav.goldberg@gmail.com)
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

from absl import logging
from common import *
from common.deps import DependenciesCollection
from common.exceptions import *

# learners
class Configuration:  # {{{

    def __init__(self, sent):
        self.stack = []
        self.sent = sent
        self.deps = DependenciesCollection()
        self.i = 0

        self.actions = []

        self._action_scores = []

    def __deepcopy__(self, memo):
        # @@ TODO: how to create the proper Configuration in the derived class?
        c = self.__class__(self.sent)
        c.deps = copy.deepcopy(self.deps, memo)
        c.i = self.i
        c.stack = self.stack[:]
        c.actions = self.actions[:]
        return c

    def actions_map(self):
        """
        returns:
           a dictionary of ACTION -> function_name. to be provided in the derived class
           see ArcStandardConfiguration for an example
        """
        return {}

    def score(self, action): pass  # @TODO

    def is_in_finish_state(self):
        logging.debug("is_in_finish_state i: %s, sent len: %s", self.i, len(self.sent))
        return len(self.stack) == 1 and not self.sent[self.i:]

    def do_action(self, action):
        logging.debug("do action: %s| i: %s | stack: %s", action, self.i, self.stack)
        return self.actions_map()[action]()

    def newAfter(self, action):
        """
        return a new configuration based on current one after aplication of ACTION
        """
        conf = copy.deepcopy(self)
        conf.do_action(action)
        return conf

class ArcStandardConfiguration(Configuration):  # {{{
    def actions_map(self):
        return {
            SHIFT: self.do_shift,
            REDUCE_R: self.do_reduceR,
            REDUCE_L: self.do_reduceL}

    def do_shift(self):
        logging.debug("ArcStandardConfiguration do_shift")
        
        if not (self.sent[self.i:]):
            logging.debug("ArcStandardConfiguration raising error ")
            raise IllegalActionException()
        self.actions.append(SHIFT)
        self._features = []
        self.stack.append(self.sent[self.i])
        self.i += 1

    def do_reduceR(self):
        logging.debug("ArcStandardConfiguration do_reduceR")
        if len(self.stack) < 2:
            print("ArcStandardConfiguration do_reduceR error.")
            raise IllegalActionException()
        self.actions.append(REDUCE_R)
        self._features = []
        stack = self.stack
        deps = self.deps

        tokt = stack.pop()  # tok_t
        tokt1 = stack.pop()  # tok_t-1
        deps.add(tokt1, tokt)
        stack.append(tokt1)

    def do_reduceL(self):
        logging.debug("ArcStandardConfiguration do_reduceL")
        if len(self.stack) < 2:
            print("ArcStandardConfiguration do_reduceL error.")
            raise IllegalActionException()
        self.actions.append(REDUCE_L)
        self._features = []
        stack = self.stack
        deps = self.deps

        tokt = stack.pop()  # tok_t
        tokt1 = stack.pop()  # tok_t-1
        deps.add(tokt, tokt1)
        stack.append(tokt)

    def valid_actions(self):
        res = []
        if self.sent[self.i:]:
            res.append(SHIFT)
        if len(self.stack) >= 2:
            res.append(REDUCE_L)
            res.append(REDUCE_R)
        return res


class ArcEagerConfiguration(Configuration):  # {{{
    """
    Nivre's ArcEager parsing algorithm
    with slightly different action names:

       Nivre's        ThisCode
       ========================
       SHIFT          SHIFT
       ARC_L          REDUCE_L
       ARC_R          REDUCE_R
       REDUCE         POP

    """

    def is_in_finish_state(self):
        return not self.sent[self.i:]

    def actions_map(self):
        return {
            SHIFT: self.do_shift,
            REDUCE_R: self.do_reduceR,
            REDUCE_L: self.do_reduceL,
            POP: self.do_pop}

    def do_shift(self):
        logging.debug("do_shift")
        if not (self.sent[self.i:]):
            raise IllegalActionException()
        self.actions.append(SHIFT)
        self._features = []
        self.stack.append(self.sent[self.i])
        self.i += 1

    def do_reduceR(self):
        logging.debug("do_reduceR")
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_R)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # attach the tokens, keeping having both on the stack
        parent = stack[-1]
        child = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        deps.add(parent, child)
        self.stack.append(child)
        self.i += 1

    def do_reduceL(self):
        logging.debug("do_reduceL")
        if len(self.stack) < 1:
            raise IllegalActionException()
        if len(self.sent) <= self.i:
            raise IllegalActionException()
        self.actions.append(REDUCE_L)
        self._features = []
        stack = self.stack
        deps = self.deps
        sent = self.sent

        # add top-of-stack as child of sent, pop stack
        child = stack[-1]
        parent = sent[self.i]
        if deps.has_parent(child):
            raise IllegalActionException()
        stack.pop()
        deps.add(parent, child)

    def do_pop(self):
        stack = self.stack

        if len(stack) == 0:
            raise IllegalActionException()
        # also illegal to pop when the item to be popped does not have a
        # parent. (can this happen? yes, right after a shift..)
        if not self.deps.has_parent(stack[-1]):
            if stack[-1]['parent'] != -1:
                raise IllegalActionException()

        self.actions.append(POP)
        self._features = []

        stack.pop()

    def valid_actions(self):
        res = [SHIFT, REDUCE_R, REDUCE_L, POP]

        if not (self.sent[self.i:]):
            res.remove(SHIFT)

        if len(self.stack) == 0:
            res.remove(POP)
        elif not self.deps.has_parent(self.stack[-1]):
            res.remove(POP)

        if len(self.stack) < 1:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        elif len(self.sent) <= self.i:
            res.remove(REDUCE_L)
            res.remove(REDUCE_R)
        else:
            if self.deps.has_parent(self.stack[-1]):
                res.remove(REDUCE_L)
            if self.deps.has_parent(self.sent[self.i]):
                res.remove(REDUCE_R)

        return res