#!/usr/bin/python3
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import time
import tensorflow as tf
from tensorflow import keras


# 位置编码
def get_angles(pos, i, d_model):
    """
    求角度
    pos.shape: [setence_length, 1]
    i.shape: [1, d_model]
    result.shape: [setence_length, d_model]
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def get_position_embedding(setence_length, d_model):
    """求编码"""
    angle_rads = get_angles(np.arange(setence_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :], d_model)

    # angle_rads.shape: [setence_length, d_model / 2]
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # position_embedding.shape: [1, setence_length, d_model]
    position_embedding = angle_rads[np.newaxis, ...]
    return tf.cast(position_embedding, dtype=tf.float32)


def plot_position_embedding(position_embedding):
    """画出位置编码的图"""
    plt.pcolormesh(position_embedding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


# mask构建
def create_padding_mask(batch_data):
    """
    padding mask
    batch_data.shape: [batch_size, seq_len]
    """
    padding_mask = tf.cast(tf.math.equal(batch_data, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    return padding_mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    look ahead只能看到前面的，看不到后面的
    下三角都是0，上三角都是1
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # shape: [seq_len, seq_len]
    return mask


# mask 总体构建
def create_masks(inp, tar):
    """
    Encoder:
      - encoder_padding_mask (self attention of EncoderLayer)
    Decoder:
      - look_ahead_mask (self attention of DecoderLayer)
      - encoder_decoder_padding_mask (encoder-decoder attention of DecoderLayer)
      - decoder_padding_mask (self attention of DecoderLayer)
    """
    encoder_padding_mask = create_padding_mask(inp)
    encoder_decoder_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decoder_padding_mask = create_padding_mask(tar)
    decoder_mask = tf.maximum(decoder_padding_mask, look_ahead_mask)

    #     print(encoder_padding_mask.shape)
    #     print(encoder_decoder_padding_mask.shape)
    #     print(look_ahead_mask.shape)
    #     print(decoder_padding_mask.shape)
    #     print(decoder_mask.shape)
    return encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask


# 缩放点积注意力机制
def scaled_dot_product_attention(q, k, v, mask):
    """
    Args:
    - q: shape == (..., seq_len_q, depth)
    - k: shape == (..., seq_len_k, depth)
    - v: shape == (..., seq_len_v, depth_v)
    - seq_len_k == seq_len_v
    - mask: shape == (..., seq_len_q, seq_len_k) 默认为None
    Returns:
    - output: weighted sum
    - attention_weights: weights of attention
    """
    # matmul_qk.hape: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        # 目的是使得在softmax后，值趋近于0
        scaled_attention_logits += (mask * -1e9)

    # attention_weights.shape: [..., seq_len_q, seq_len_k]
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 加权求和
    # output.hape: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


# 打印缩放点积，方便调试
def print_scaled_dot_product_attention(q, k, v):
    temp_out, temp_att = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_att)
    print('Output is:')
    print(temp_out)
