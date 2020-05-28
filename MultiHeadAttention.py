#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from utils import *

# 多头注意力
class MultiHeadAttention(keras.layers.Layer):
    """
    理论上：
    x -> Wq0 -> q0
    x -> Wk0 -> k0
    x -> Wv0 -> v0
    实际上：把x分成q,k,v
    q -> Wq0 -> q0
    k -> Wk0 -> k0
    v -> Wv0 -> v0
    实战中的技巧：
    q -> Wq -> Q -> split -> q0,q1,q2...
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.WQ = keras.layers.Dense(self.d_model)
        self.WK = keras.layers.Dense(self.d_model)
        self.WV = keras.layers.Dense(self.d_model)

        self.dense = keras.layers.Dense(self.d_model)

    def split_heads(self, x, batch_size):
        # x.shape: (batch_size, seq_len, d_model)
        # d_model = num_heads * depth
        # x -> (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.WQ(q)  # q.shape: (batch_size, seq_len_q, d_model)
        k = self.WK(k)  # k.shape: (batch_size, seq_len_k, d_model)
        v = self.WV(v)  # v.shape: (batch_size, seq_len_v, d_model)

        q = self.split_heads(q, batch_size)  # q.shape: (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # k.shape: (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # v.shape: (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention_outputs.shape: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention_outputs, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # scaled_attention_outputs.shape: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention_outputs = tf.transpose(scaled_attention_outputs, perm=[0, 2, 1, 3])

        # concat_attention.shape: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention_outputs, (batch_size, -1, self.d_model))

        # output.shape: (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)
        return output, attention_weights
