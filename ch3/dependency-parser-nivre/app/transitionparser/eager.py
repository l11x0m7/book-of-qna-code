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
sys.path.insert(0, os.path.join(curdir, os.path.pardir))

if sys.version_info[0] < 3:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    raise "Must be using Python 3"

from absl import app
from absl import flags
from absl import logging

from ml import ml
from pio import io
from transitionparser.oracles import *
from transitionparser.deciders import *
from transitionparser.parsers import *
from features import extractors
from common.utils import is_projective

FLAGS = flags.FLAGS
'''
General
'''

flags.DEFINE_boolean('ignore_punc', False, 'Ignore Punct File.')  # nopep8
flags.DEFINE_boolean('only_projective', False, 'Only Projective.')  # nopep8
flags.DEFINE_boolean('lazypop', True, 'Lazy pop.')   # nopep8
flags.DEFINE_boolean('unlex', False, 'unlex')   # nopep8
flags.DEFINE_string('feature_extarctor', 'eager.zhang', 'Feature Extarctor')  # nopep8
flags.DEFINE_string('model', os.path.join(curdir, os.path.pardir, os.path.pardir, "tmp", "eager.model"), 'Transition Parser Model.')  # nopep8

'''
Train
'''
flags.DEFINE_boolean('train', False, 'Train model with train data')  # nopep8
flags.DEFINE_integer('epoch', 1, 'Train Epoch.')  # nopep8
flags.DEFINE_string('train_data', os.path.join(curdir, os.path.pardir, os.path.pardir, "data", "conll.example"), 'Train Data')  # nopep8
flags.DEFINE_string('externaltrainfile', None, 'External Train File.')  # nopep8
# flags.DEFINE_string('modelfile', 'data/weights', 'Model File.')

'''
Test
'''
flags.DEFINE_boolean('test', False, 'Evalutate with test data')  # nopep8
flags.DEFINE_string('test_data', os.path.join(curdir, os.path.pardir, os.path.pardir, "data", "conll.example"), 'Test data.')  # nopep8
flags.DEFINE_string('test_results', os.path.join(curdir, os.path.pardir, os.path.pardir, "tmp", "eager.test.results"), 'Save scores into disk.')  # nopep8



def test():
    '''
    Test Model
    '''
    logging.info("test ...")
    featExt = extractors.get(FLAGS.feature_extarctor)
    p = ArcEagerParser(
        MLActionDecider(
            ml.MulticlassModel(
                FLAGS.model),
            featExt))

    good = 0.0
    bad = 0.0
    complete = 0.0

    # main test loop
    reals = set()
    preds = set()
    with open(FLAGS.test_results, "w") as fout:
        sents = io.transform_conll_sents(FLAGS.test_data, FLAGS.only_projective, FLAGS.unlex)
        for i, sent in enumerate(sents):
            sgood = 0.0
            sbad = 0.0
            mistake = False
            sys.stderr.write("%s %s %s\n" %
                             ("@@@", i, good / (good + bad + 1)))
            try:
                d = p.parse(sent)
            except MLTrainerWrongActionException:
                # this happens only in "early update" parsers, and then we just go on to
                # the next sentence..
                continue
            sent = d.annotate_allow_none(sent)
            for tok in sent:
                if FLAGS.ignore_punc and tok['form'][0] in "`',.-;:!?{}":
                    continue
                reals.add((i, tok['parent'], tok['id']))
                preds.add((i, tok['pparent'], tok['id']))
                if tok['pparent'] == -1:
                    continue
                if tok['parent'] == tok['pparent'] or tok['pparent'] == -1:
                    good += 1
                    sgood += 1
                else:
                    bad += 1
                    sbad += 1
                    mistake = True
            if FLAGS.unlex:
                io.out_conll(sent, parent='pparent', form='oform')
            else:
                io.out_conll(sent, parent='pparent', form='form')
            if not mistake:
                complete += 1
            # sys.exit()
            logging.info("test result: sgood[%s], sbad[%s]", sgood, sbad)
            if sgood > 0.0 and sbad > 0.0:
                fout.write("%s\n" % (sgood / (sgood + sbad)))

        logging.info("accuracy: %s", good / (good + bad))
        logging.info("complete: %s", complete / len(sents))
        preds = set([(i, p, c) for i, p, c in preds if p != -1])
        logging.info("recall: %s", len(
            preds.intersection(reals)) / float(len(reals)))
        logging.info("precision: %s", len(
            preds.intersection(reals)) / float(len(preds)))
        logging.info("assigned: %s", len(preds) / float(len(reals)))


def train():
    '''
    Train Model
    '''
    MODE = 'train'
    TRAIN_OUT_FILE = FLAGS.model

    if FLAGS.externaltrainfile:
        '''
        create feature vector files for training with an external classifier.  If you don't know what it means,
         just ignore this option.  The model file format is the same as Megam's.
        '''
        MODE = 'write'
        TRAIN_OUT_FILE = FLAGS.externaltrainfile

    featExt = extractors.get(FLAGS.feature_extarctor)
    sents = io.transform_conll_sents(FLAGS.train_data, FLAGS.only_projective, FLAGS.unlex)

    if MODE == "write":
        fout = file(TRAIN_OUT_FILE, "w")
        trainer = LoggingActionDecider(
            ArcEagerParsingOracle(
                pop_when_can=FLAGS.lazypop), featExt, fout)
        p = ArcEagerParser(trainer)
        for i, sent in enumerate(sents):
            sys.stderr.write(". %s " % i)
            sys.stderr.flush()
            d = p.parse(sent)
        sys.exit()

    if MODE == "train":
        fout = open(TRAIN_OUT_FILE, "w")
        nactions = 4
        trainer = MLTrainerActionDecider(
            ml.MultitronParameters(nactions), ArcEagerParsingOracle(
                pop_when_can=FLAGS.lazypop), featExt)
        p = ArcEagerParser(trainer)
        import random
        random.seed("seed")
        total = len(sents)
        for x in range(FLAGS.epoch):  # epoch
            logging.info("iter %s/%s", x + 1, FLAGS.epoch)
            logging.info("  shuffle data ...")
            random.shuffle(sents)
            for i, sent in enumerate(sents):
                if i % 500 == 0:
                    logging.info("  step %s/%s ...", i, total)
                try:
                    d = p.parse(sent)
                except IndexError as e:
                    logging.info("prob in sent: %s", i)
                    logging.info("\n".join(
                        ["%s %s %s %s" % (t['id'], t['form'], t['tag'], t['parent']) for t in sent]))
                    raise e
        logging.info("save model file to disk [%s] ...", TRAIN_OUT_FILE)
        trainer.save(fout)


def main(argv):
    print(
        'Running under Python {0[0]}.{0[1]}.{0[2]}'.format(
            sys.version_info),
        file=sys.stderr)
    if FLAGS.train:
        train()
    if FLAGS.test:
        test()


if __name__ == '__main__':
    # FLAGS([__file__, '--verbosity', '1'])
    app.run(main)
