# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

VERY_NEGATIVE_NUMBER = -1e30

"""
这个文件里主要是定义一些基本的模块，相对于tf版本的layer,
在pytorch里我们继承module
"""


class Highway(nn.Module):
    def __init__(self,
                 affine_activation=F.relu,
                 trans_gate_activation=F.sigmoid,
                 hidden_units=0,
                 keep_prob=1.0,
                 num_layers=1):
        super(Highway, self).__init__()
        self.affine_activation = affine_activation
        self.trans_gate_activation = trans_gate_activation
        self.affine_layer = nn.ModuleList([nn.Linear(hidden_units, hidden_units)
                                          for _ in range(num_layers)])
        self.trans_gate_layer = nn.ModuleList([nn.Linear(hidden_units, hidden_units)
                                               for _ in range(num_layers)])

    def forward(self, x, training=True):
        gate = self.affine_activation(self.trans_gate_layer(x))
        # TODO dropout
        trans = self.trans_gate_activation(self.affine_layer(x))

        return gate * trans + (1. - gate) * x


class Embedding(nn.Module):
    def __init__(self, pretrained_embedding=None,
                 embedding_shape=None,
                 trainable=True,
                 init_scale=0.02):
        super(Embedding, self).__init__()
        if pretrained_embedding is None and embedding_shape is None:
            raise ValueError("At least one of pretrained_embedding and embedding_shape must be specified!")

        if pretrained_embedding is None:
            # \mathcal{N}(-init_scale, init_scale)
            self.embedding = nn.Embedding(embedding_shape[0], embedding_shape[1])
            nn.init.uniform_(self.embedding.weight, -init_scale, init_scale)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding)

        if not trainable:
            # do not update the weight
            self.embedding.weight.requires_grad = False

    def forward(self, indices):
        return self.embedding(indices)


class Conv1DAndMaxPooling(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, strides=1,  padding=0, activation=F.relu):
        super(Conv1DAndMaxPooling, self).__init__()
        self.conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, strides)

    def forward(self, x, seq_len=None):
        input_shape = x.size()
        if len(input_shape) == 4:
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            x = x.view(-1, input_shape[-2], input_shape[-1])
            x = self.conv_layer(x)
            if seq_len is not None:
                hidden_units = x.size(-1)
                x = x.view(batch_size, seq_length, x.size(1), hidden_units)
                x = self.max_pooling(x, seq_len)
            else:
                x = x.max(1)
                x = x.view(batch_size, -1, x.size(-1))
        elif len(input_shape) == 3:
            x = self.conv_layer(x)
            x = x.max(1)
        else:
            raise ValueError()

        return x

    def max_pooling(self, inputs, seq_mask=None):
        """

        :param inputs:
        :param seq_mask: [batch, seq_len]
        :return:
        """
        rank = len(inputs.size())-2
        if seq_mask is not None:
            size = inputs.size()
            seq_mask = seq_mask.type(torch.float32).view(size(0), size(1), size(2), 1)
            inputs = inputs*seq_mask+(1.-seq_mask)*VERY_NEGATIVE_NUMBER

        return inputs.max(rank)
