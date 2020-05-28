#!/usr/bin/python3
# -*- coding:utf-8 -*-
from tensorflow import keras


# feed_forward
def feed_forward_network(d_model, dff):
    # dff: dim of feed forward network.
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),
        keras.layers.Dense(d_model)
    ])
