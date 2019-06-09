# coding:utf-8
import logging
import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch.nn as nn
import torch

"""
基本模型框架：
1. load
2. save
3. train and eval
4. eval
5. inference
6. get best answer
"""


class BaseModel(nn.Module):
    def __init__(self, vocab=None):
        super(BaseModel, self).__init__()
        self.vocab = vocab

        self.initialized = False
        self.ema_decay = 0
        self.optimizer = None

    def __del__(self):
        # todo
        pass

    def load(self, path):
        logging.info('Loading model from %s' % path)
        # todo
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def save(self,loss, path, epoch, global_step=None):
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        # torch.save(model_to_save.state_dict(), path)
        logging.info('Saving model to %s' % path)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss
            },
            path
        )

    def forward(self, *input):
        raise NotImplementedError

    def update(self, *input):
        """
        update weight and bias
        :param input:
        :return:
        """
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            raise Exception("The model need to compile!")

    def compile(self, *input):
        """
        set the optimizer
        :param input:
        :return:
        """
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
