from collections import defaultdict
import os
import torch
import logging
import numpy as np


class Trainer(object):
    def __init__(self):
        pass

    @staticmethod
    def _train():
        raise NotImplementedError

    @staticmethod
    def _eval():
        raise NotImplementedError

    @staticmethod
    def _train_and_evaluate(model, train_batch_generator, eval_batch_generator, evaluator, epochs=1, eposides=1, save_dir=None):
        # prepare some summary and saver TODO
        best_eval_score = 0.0

        for epoch in range(epochs):
            logging.info("Epoch {}/{}".format(epoch + 1, epochs))
            # compute the train_num_steps
            train_num_steps = 0

            assert isinstance(eposides, int)
            num_steps_per_eposide = (train_num_steps + eposides - 1) // eposides
            for eposide in range(eposides):
                logging.info("Eposide {}/{}".format(eposide + 1, eposides))
                current_step_num = min(num_steps_per_eposide, train_num_steps - eposide * num_steps_per_eposide)
                eposide_id = epoch * eposides + eposide + 1
                # train once
                Trainer._train()

                if model.ema_decay > 0:
                    # 滑动平均
                    continue
                if save_dir is not None:
                    last_save_path = os.path.join(save_dir, 'last_weights', 'after-eposide')
                    # save model
                    # global_step = eposide_id

                # evaluation
                # 和原有的不同的时，不是每个step都eval
                eval_instances = eval_batch_generator.get_instances()
                output = Trainer._eval()

                score = evaluator.get_score(model.get_best_answer(output, eval_instances))

                # 根据score save best weights

        raise NotImplementedError

    @staticmethod
    def inference():
        raise NotImplementedError

    @staticmethod
    def _evaluate():
        raise NotImplementedError

    @staticmethod
    def _inference():
        raise NotImplementedError