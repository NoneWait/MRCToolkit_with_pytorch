from collections import defaultdict
import os
import torch
import logging
import numpy as np
import torch.utils.data.dataloader


class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train(model, batch_generator, n_gpu, device):
        # global_step = None
        for batch in batch_generator:
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            # batch 是以tuple给定的, 所以model要自行在输入中将其展开
            # compute the loss
            loss = model(batch)
            if n_gpu > 1:
                loss = loss.mean()
            # compute grad
            loss.backward()
            # update the weight and bias
            model.update()

    @staticmethod
    def _eval():
        raise NotImplementedError

    @staticmethod
    def _train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, n_gpu, device,
                            epochs=1, save_dir=None):
        # prepare some summary and saver TODO
        best_eval_score = 0.0

        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            # set mode to train
            model.train()
            Trainer._train(model, train_batch_generator, n_gpu, device)

            # model evaluation
            model.eval()
            Trainer._eval()

            # TODO model save

    @staticmethod
    def inference():
        raise NotImplementedError

    @staticmethod
    def _evaluate():
        raise NotImplementedError

    @staticmethod
    def _inference():
        raise NotImplementedError
