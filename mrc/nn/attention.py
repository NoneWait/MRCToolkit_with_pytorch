# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

VERY_NEGATIVE_NUMBER = -1e30


class BiAttention(nn.Module):
    """ Bi-Directonal Attention from https://arxiv.org/abs/1611.01603 """

    def __init__(self, similarity_function):
        super(BiAttention, self).__init__()
        self.similarity_function = similarity_function

    def forward(self, query, memory, query_mask, memory_mask):
        """
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        :param query: [batch, n, d]
        :param memory: [batch, m, d]
        :param query_mask:
        :param memory_mask:
        :return:
        """
        sim_mat = self.similarity_function(query, memory)
        batch_size, q_len, _ = query.size()
        m_len = memory.size(1)
        q_mask = query_mask.view(batch_size, q_len, 1)
        m_mask = memory_mask.view(batch_size, 1, m_len)
        mask = q_mask * m_mask
        sim_mat = sim_mat + (1. - mask) * VERY_NEGATIVE_NUMBER
        # context-to-query attention
        query_memory_prob = F.softmax(sim_mat)
        query_memory_attention = torch.bmm(query_memory_prob, memory)

        # query-to-context attention
        # [batch, n]
        memory_query_prob = F.softmax(sim_mat.max(-1))
        # [batch, 1, d]
        memory_query_attention = torch.bmm(memory_query_prob.unsqueeze(1), query)
        memory_query_attention = memory_query_attention.repeat(1, q_len, 1)

        return query_memory_attention, memory_query_attention
