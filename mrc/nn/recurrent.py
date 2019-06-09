# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


"""
主要封装处理变长序列的rnn
"""


class BiLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 drop_prob=0.):
        super(BiLSTM, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           drop_prob=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        orig_len = x.size(1)

        # pack
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # apply rnn
        x, _ = self.rnn(x)

        # unpack
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]

        # apply dropout
        x = F.dropout(x, self.drop_prob, self.training)

        return x
