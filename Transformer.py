#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from EncoderModel import EncoderModel
from DecoderModel import DecoderModel


class Transformer(keras.Model):
    def __init__(self, num_layers, input_vocab_size, target_vocab_size, max_length,
                 d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder_model = EncoderModel(
            num_layers, input_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.decoder_model = DecoderModel(
            num_layers, target_vocab_size, max_length, d_model, num_heads, dff, rate)
        self.final_layer = keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, encoder_padding_mask,
             decoder_mask, encoder_decoder_padding_mask):
        # encoding_outputs.shape: (batch_size, input_seq_len, d_model)
        encoding_outputs = self.encoder_model(inp, training, encoder_padding_mask)

        # decoding_outputs.shape: (batch_size, output_seq_len, d_model)
        decoding_outputs, attention_weights = self.decoder_model(
            tar, encoding_outputs, training, decoder_mask, encoder_decoder_padding_mask)

        # decoding_outputs.shape: (batch_size, output_seq_len, target_vocab_size)
        predictions = self.final_layer(decoding_outputs)

        return predictions, attention_weights
