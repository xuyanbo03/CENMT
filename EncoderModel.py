#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow
from tensorflow import keras
from utils import *
from EncoderLayer import EncoderLayer


class EncoderModel(keras.layers.Layer):
    def __init__(self, num_layers, input_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(EncoderModel, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = keras.layers.Embedding(input_vocab_size, self.d_model)
        # position_embedding.shape: (1, max_length, d_model)
        self.position_embedding = get_position_embedding(self.max_length, self.d_model)
        self.dropout = keras.layers.Dropout(rate)
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]

    def call(self, x, training, encoder_padding_mask):
        # x.shape: (batch_size, input_seq_len)
        input_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(
            input_seq_len, self.max_length,
            message='input_seq_len should be less or equal to self.max_length')

        # x.shape: (batch_size, input_seq_len, d_model)
        x = self.embedding(x)
        # 做缩放，范围是0-d_model，目的是在与position_embedding做完加法后，x起的作用更大
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding[:, :input_seq_len, :]
        x = self.dropout(x, training=training)

        # x.shape: (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, training, encoder_padding_mask)

        return x
