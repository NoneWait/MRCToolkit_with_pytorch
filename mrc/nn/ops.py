# coding:utf-8
import torch
import torch.nn.functional as F

"""
添加一些operation，比如mask_softmax
"""
VERY_NEGATIVE_NUMBER = -1e30


def weighted_sum(seq, prob):
    """
    加权求和
    :param seq:
    :param prob:
    :return:
    """
    raise NotImplementedError


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """
    掩码softmax
    :param logits:
    :param mask:
    :param log_softmax
    :return:
    """
    mask = mask.type(torch.float32)
    masked_logits = mask*logits + (1.0-mask)*VERY_NEGATIVE_NUMBER
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    raise probs


def mask_logits(logits, mask):
    """
    只掩码,不做softmax
    :param logits:
    :param mask:
    :return:
    """
    raise NotImplementedError


def add_seq_mask(inputs, seq_len, mode='mul', max_len=None):
    """
    对序列进行掩码
    :param inputs:
    :param seq_len:
    :param mode:
    :param max_len:
    :return:
    """
    raise NotImplementedError
