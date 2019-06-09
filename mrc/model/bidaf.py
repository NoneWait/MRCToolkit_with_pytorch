# coding:utf-8
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

import logging

from mrc.model.base_model import BaseModel
from mrc.nn.layers import Embedding, Highway
from mrc.nn.attention import BiAttention
from mrc.nn.similarity_function import TriLinear
from mrc.nn.recurrent import BiLSTM
from mrc.nn.ops import masked_softmax



class BiDAF(BaseModel):
    """
    https://arxiv.org/pdf/1611.01603.pdf
    """
    def __init__(self, vocab, pretrained_word_embedding=None,word_embedding_size=100, char_embedding_size=8,
                 char_conv_filters=100,
                 char_conv_kernel_size=5, rnn_hidden_size=100,
                 dropout_keep_prob=0.8, max_answer_len=17, word_embedding_trainable=False,use_elmo=False,elmo_local_path=None,
                 enable_na_answer=False,
                 training=True):
        super(BiDAF, self).__init__(vocab)
        self.rnn_hidden_size = rnn_hidden_size
        self.keep_prob = dropout_keep_prob
        self.word_embedding_size = word_embedding_size
        self.pretrained_word_embedding = pretrained_word_embedding
        self.char_embedding_size = char_embedding_size
        self.char_conv_filters = char_conv_filters
        self.char_conv_kernel_size = char_conv_kernel_size
        self.max_answer_len = max_answer_len
        self.use_elmo = use_elmo
        self.elmo_local_path=elmo_local_path
        self.word_embedding_trainable = word_embedding_trainable
        self.enable_na_answer = enable_na_answer # for squad2.0
        self.training = training

        self.word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                        embedding_shape=(len(self.vocab.get_word_vocab())+1, self.word_embedding_size),
                                        trainable=self.word_embedding_trainable)
        # TODO add char embedding
        self.char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size)
                                        , trainable=True, init_scale=0.2)

        self.highway = Highway(num_layers=2)

        self.phrase_lstm = BiLSTM(None, self.rnn_hidden_size)

        # TODO hidden_units 需要计算
        self.bi_attention = BiAttention(TriLinear(hidden_units=100))

        self.modeling_lstm = BiLSTM(None, self.rnn_hidden_size, num_layers=2)

        self.start_pred_layer = nn.Linear(None, 1, bias=False)

    def forward(self, data):
        (context_word, context_char, context_len,
        question_word, question_char, question_len,
        start_positions, end_positions,
        question_tokens, context_tokens, na) = data

        c_mask = None
        q_mask = None
        c_len = None
        q_len = None
        # 1.1 Embedding
        context_word_repr = self.word_embedding(context_word)
        context_char_repr = self.char_embedding(context_char)
        question_word_repr = self.word_embedding(question_word)
        question_char_repr = self.char_embedding(question_char)

        # 1.2 Char convolution
        # TODO

        # elmo embedding
        # TODO

        # concat
        context_repr = context_word_repr
        question_repr = question_word_repr

        # 1.3 Highway network
        context_repr = self.highway(context_repr)
        question_repr = self.highway(question_repr)

        # 2. Phrase encoding TODO
        context_repr = self.phrase_lstm(context_repr, c_len)
        question_repr = self.phrase_lstm(question_repr, q_len)

        # 3. Bi-Attention
        c2q, q2c = self.bi_attention(context_repr, question_repr, c_mask, q_mask)

        # 4.Modeling layer TODO
        # 区别于SMRC，它是将两层的输出做一个add操作，论文里不是这么实现的
        final_merged_context = torch.cat([context_repr, c2q, context_repr*c2q, context_repr*q2c], dim=-1)
        modeling_context = self.modeling_lstm(final_merged_context, c_len)
        # 5. Start prediction
        start_logits = self.start_pred_layer(torch.cat([final_merged_context, modeling_context], dim=-1))
        start_logits = start_logits.squeeze(-1)
        # 6. End prediction
        end_logits = None

        # if train return loss, if eval return start_logits and end_logits
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

    def compile(self, initial_lr):
        self.optimizer = Adam(self.parameters(), lr=initial_lr)

    # def update(self):
    #     self.optimizer.step()
    #     self.optimizer.zero_grad()
