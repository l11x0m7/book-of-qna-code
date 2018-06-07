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

from absl import logging
from collections import defaultdict
from common import *

class ArcStandardParsingOracle:  # {{{
    def __init__(self):
        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        pass

    def next_action_from_config(self, conf):
        return self.next_action(conf.stack, conf.deps, conf.sent, conf.i)

    def next_action(self, stack, deps, sent, i):
        """
        assuming sent has 'parent' information for all tokens

        need to find all childs of a token before combining it to the head
        """
        if self.current_sent != sent:
            self.current_sent = sent
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        # if stack has < 2 elements, must shift
        if len(stack) < 2:
            return SHIFT
        # else, if two items on top of stack should connect,
        # choose the correct order
        if stack[-2]['parent'] == stack[-1]['id']:
            # if found_all_childs(stack[-2]):
            if len(self.childs_of_x[stack[-2]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-2]['id'])
                return REDUCE_L
            else:
                pass
        if stack[-1]['parent'] == stack[-2]['id']:
            # if found_all_childs(stack[-1]):
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                self.connected_childs.add(stack[-1]['id'])
                return REDUCE_R
            else:
                pass
        # else
        if len(sent) <= i:
            return REDUCE_L
        return SHIFT

class ArcEagerParsingOracle:  # {{{
    def __init__(self, pop_when_can=True):
        self.current_sent = None
        self.childs_of_x = None
        self.connected_childs = set()
        self.POP_WHEN_CAN = pop_when_can
        pass

    def next_actions(self, stack, deps, sent, i):
        return [self.next_action(stack, deps, sent, i)]

    def next_action(self, stack, deps, sent, i):
        """
        if top-of-stack has a connection to next token:
           do reduce_L / reduce_R based on direction
        elsif top-of-stack-minus-1 has a connection to next token:
           do pop
        else do shift

        assuming sent has 'parent' information for all tokens

        assuming several possibilities, the order is left > right > pop > shift.

        see: "An efficient algorithm for projective dependency parsing" / Joakim Nivre, iwpt 2003

        """
        if self.current_sent != sent:
            self.current_sent = sent
            self.childs_of_x = defaultdict(set)
            self.connected_childs = set([-1])
            for tok in sent:
                self.childs_of_x[tok['parent']].add(tok['id'])

        # if stack has < 1 elements, must shift
        if len(stack) < 1:
            return SHIFT

        #ext = sent[i]['extra'].split("|")
        # if len(ext)>1 and ext[1] in ['NONDET']:
        #   return SHIFT

        if stack and sent[i:] and stack[-1]['parent'] == sent[i]['id']:
            self.connected_childs.add(stack[-1]['id'])
            return REDUCE_L

        if stack and sent[i:] and stack[-1]['id'] == sent[i]['parent']:
            if deps.has_parent(sent[i]):
                logging.debug("skipping add: stack [%s], sent [%s]", stack[-1], sent[i])
                pass
            else:
                self.connected_childs.add(sent[i]['id'])
                return REDUCE_R

        if len(stack) > 1:
            # POP when you can
            # if found_all_childs(stack[-1]):
            if len(self.childs_of_x[stack[-1]['id']] -
                   self.connected_childs) < 1:
                if deps.has_parent(stack[-1]):
                    if self.POP_WHEN_CAN:
                        return POP
                    else:  # pop when must..
                        # go up to parents. if a parent has a right-child,
                        # then we need to reduce in order to be able to build it.
                        # else, why reduce?
                        par = deps.parent(stack[-1])
                        while par is not None:
                            par_childs = self.childs_of_x[par['id']]
                            for c in par_childs:
                                if c > stack[-1]['id']:
                                    return POP
                            # if the paren't parent is on the right --
                            # we also need to reduce..
                            if par['parent'] > stack[-1]['id']:
                                return POP
                            par = deps.parent(par)
                        # if we are out of the loop: no need to reduce
        if i < len(sent):
            logging.debug("defaulting: to shift..")
            return SHIFT

        assert(False)
