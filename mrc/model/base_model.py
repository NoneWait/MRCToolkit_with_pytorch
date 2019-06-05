# coding:utf-8
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict


class BaseModel(object):
    def __init__(self, vocab=None):
        self.vocab = vocab

        self.initialized = False
        self.ema_decay = 0

    def __del__(self):
        # todo
        pass

    def load(self, path, var_list=None):
        logging.info('Loading model from %s' % path)
        # todo
        self.initialized = True

    def save(self, path, global_step=None, var_list=None):
        pass

    def _build_graph(self):
        raise NotImplementedError

    def compile(self, *input):
        raise NotImplementedError

    def get_best_answer(self, *input):
        raise NotImplementedError

    def train_and_evaluate(self, train_generator, eval_generator, evaluator, epochs=1, eposides=1,
                           save_dir=None, summary_dir=None, save_summary_steps=10):
        pass

    def evaluate(self, batch_generator, evaluator):
        pass

    def inference(self, batch_generator):
        pass
