# coding:utf-8
import torch
import logging

from mrc.model.base_model import BaseModel
from mrc.nn.layers import Embedding, Highway
from mrc.nn.attention import BiAttention
from mrc.nn.similarity_function import TriLinear


class BiDAF(BaseModel):
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
        self.elmo_local_path= elmo_local_path
        self.word_embedding_trainable = word_embedding_trainable
        self.enable_na_answer = enable_na_answer # for squad2.0
        self.training = training

        self.word_embedding = Embedding(pretrained_embedding=self.pretrained_word_embedding,
                                        embedding_shape=(len(self.vocab.get_word_vocab())+1, self.word_embedding_size),
                                        trainable=self.word_embedding_trainable)
        # TODO add char embedding
        self.char_embedding = Embedding(embedding_shape=(len(self.vocab.get_char_vocab()) + 1, self.char_embedding_size),
                                   trainable=True, init_scale=0.2)

        self.highway = Highway(num_layers=2)

        self.phrase_lstm = None

        # TODO hidden_units 需要计算
        self.bi_attention = BiAttention(TriLinear(hidden_units=100))


    def forward(self, context_word, context_char, context_len,
                question_word, question_char, question_len,
                answer_start, answer_len,
                question_tokens, context_tokens,na):

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
        context_repr, _ = self.phrase_lstm(context_repr)
        question_repr, _ = self.phrase_lstm(question_repr)

        # 3. Bi-Attention
        c2q, q2c = self.bi_attention(context_repr, question_repr, c_mask, q_mask)

        # 4.Modeling layer TODO

        # 5. Start prediction
        start_prob = None
        # 6. End prediction
        end_prob = None

        return start_prob, end_prob
