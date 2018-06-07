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
from transitionparser.configurations import *

class TransitionBasedParser:
    """
    Refactored ArcStandardParser, with a Configuration object
    """
    Configuration = None
    """
   Configuration class, defines how the parser behaves
   """

    def __init__(self, decider):
        self.d = decider
        pass

    def decide(self, conf):
        actions = self.d.next_actions(
            conf.stack, conf.deps, conf.sent, conf.i, conf)
        return actions

    def parse(self, sent):
        logging.debug("parse: %s", sent)
        sent = [ROOT] + sent
        conf = self.Configuration(sent)
        logging.debug("parse: resolve conf")
        while not conf.is_in_finish_state():
            logging.debug("parse: not finish")
            next_actions = self.decide(conf)
            for act in next_actions:
                try:
                    logging.debug("parse: next_actions [%s]", act)
                    conf.do_action(act)
                    logging.debug("parse: stack len(%s), deps(%s), buffer len(%s)", len(conf.stack), len(conf.deps), (len(conf.sent) - conf.i))
                    break
                except IllegalActionException as e:
                    logging.debug("parse: next_actions error - %s", e)
                    pass
        logging.debug("parse: finish")
        return conf.deps  # ,conf.chunks

class ArcStandardParser2(TransitionBasedParser):
    Configuration = ArcStandardConfiguration


class ArcEagerParser(TransitionBasedParser):
    Configuration = ArcEagerConfiguration


class ErrorInspectionParser(ArcStandardParser2):  # {{{
    def __init__(self, decider, oracle, confPrinter, out=sys.stdout):
        ArcStandardParser2.__init__(self, decider)
        self.oracle = oracle
        self.confPrinter = confPrinter
        self.out = out

        self.raise_on_error = False
        self.use_oracle_answer = True

    def decide(self, conf):
        action = ArcStandardParser2.decide(self, conf)
        real = self.oracle.next_action(
            conf.stack, conf.deps, conf.sent, conf.i)
        if action != real:
            self.out.write(
                "%s -> %s %s\n" %
                (real, action, self.confPrinter.format(conf)))
            if self.raise_on_error:
                raise MLTrainerWrongActionException()

        if self.use_oracle_answer:
            return real
        else:
            return action
