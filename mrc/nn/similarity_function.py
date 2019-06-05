# coding:utf-8
import torch
import torch.nn as nn
import math

"""
提供计算相似度矩阵计算的module
"""


class DotProduct(nn.Module):
    """
    ref: https://github.com/siat-nlp/transformer-pytorch/blob/master/transformer/Modules.py
    s = t0*t1
    """
    def __init__(self, scale=False):
        super(DotProduct, self).__init__()
        # 缩放因子
        self.scale = scale

    def forward(self, t0, t1):
        """

        :param t0: [batch, c_len, d]
        :param t1: [batch, q_len, d]
        :return:
        """
        dots = torch.bmm(t0, t1.transpose(1, 2))
        if self.scale:
            last_dims = t0.size()[-1]
            dots = dots / math.sqrt(last_dims)
        return dots


class ProjectedDotProduct(nn.Module):
    """
    math:`(x*W)*(y*W)`
    """
    def __init__(self, t0_len, t1_len, hidden_units, activation=None):
        """

        :param hidden_units:
        :param activation: function like F.relu

        """
        super(ProjectedDotProduct, self).__init__()
        self.activation = activation
        self.projecting_layer = nn.Linear(t0_len, hidden_units, bias=False)
        self.projecting_layer2 = nn.Linear(t1_len, hidden_units, bias=False)

    def forward(self, t0, t1):
        """

        :param t0: [batch,n,d]
        :param t1: [batch,m,d]
        :return: [batch, n, m]
        """
        t0 = self.projecting_layer(t0) if self.activation is None else self.activation(self.projecting_layer(t0))
        t1 = self.projecting_layer2(t1) if self.activation is None else self.activation(self.projecting_layer(t1))

        return torch.bmm(t0, t1.transpose(1, 2))


class BiLinear(nn.Module):
    """
    sim = xAy
    双线性
    """
    def __init__(self, hidden_units):
        super(BiLinear, self).__init__()
        self.projecting_layer = nn.Linear(hidden_units, hidden_units)

    def forward(self, t0, t1):
        """

        :param t0: [batch,n,d]
        :param t1: [batch,m,d]
        :return: [batch,n,m]
        """
        t0 = self.projecting_layer(t0)
        return torch.bmm(t0, t1.transpose(1, 2))


class TriLinear(nn.Module):
    """
    三线性
    """
    def __init__(self, hidden_units, bias=False):
        super(TriLinear, self).__init__()
        self.projecting_layer = nn.Linear(hidden_units, 1, bias=False)
        self.bias = bias

    def forward(self, *input):
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


class SymmetricProject(nn.Module):
    def __init__(self):
        super(SymmetricProject, self).__init__()

    def forward(self, *input):
        raise NotImplementedError



