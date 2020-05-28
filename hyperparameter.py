#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow import keras
from Transformer import Transformer
from load_data import input_tensor, input_tokenizer, output_tokenizer

# data参数
buffer_size = 10000 # 20000
batch_size = 32 # 64
max_length = 40 # 160

# 定义超参数
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
epochs = 100
steps_per_epoch = len(input_tensor) // batch_size

input_vocab_size = len(input_tokenizer.word_index) + 1
target_vocab_size = len(output_tokenizer.word_index) + 1
print(input_vocab_size, target_vocab_size)


# 自定义学习率
class CustomizedSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomizedSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** (-1.5))
        arg3 = tf.math.rsqrt(tf.cast(self.d_model, dtype=tf.float32))
        return arg3 * tf.math.minimum(arg1, arg2)


learning_rate = CustomizedSchedule(d_model)

# model
transformer = Transformer(num_layers, input_vocab_size, target_vocab_size,
                          max_length, d_model, num_heads, dff, dropout_rate)

# optimizer
optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# ckpt save
checkpoint_path = os.path.join('transformer-cmn')  # transformer-zh
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)
