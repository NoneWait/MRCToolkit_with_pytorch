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
    math:`w^T[c,q,c\circ q]`
    其中 w^T*(c\circ q) = (w\circ c)^T * q
    """
    def __init__(self, hidden_units, bias=False):
        super(TriLinear, self).__init__()
        self.projecting_layers = [nn.Linear(hidden_units, 1, bias=False)
                                  for _ in range(2)]
        self.dot_w = nn.Parameter(torch.zeros(1, 1, hidden_units))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.bias = None

        for weigth in (self.projecting_layers[0].weight, self.projecting_layers[1].weight,
                       self.dot_w):
            nn.init.xavier_uniform_(weigth)

    def forward(self, t0, t1):
        t0_len, t1_len = t0.size(1), t1.size(1)
        # [b, n, 1]
        t0_score = self.projecting_layers[0](t0)
        # [b, 1, m]
        t1_score = self.projecting_layers[1](t1).transpose(2, 1)

        # [w1*t0_1, ..., w_d*t0_d]
        # [batch, n, d] \circ [1, 1 , d] =>[batch, n, d]
        t0_dot_w = t0*self.dot_w
        # [batch, n, d]*[batch, d, m] => [batch, n, m]
        t0_t1_score = torch.matmul(t0_dot_w, t1.transpose(2, 1))

        out = t0_t1_score + t0_score.expand([-1, -1, t1_len]) + t1_score.expand([-1, t0_len, -1])

        if self.bias is not None:
            out += self.bias
        return out


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



