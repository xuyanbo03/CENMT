#!/usr/bin/python3
# -*- coding:utf-8 -*-
from utils import *
from load_data import train_dataset, eval_dataset
from hyperparameter import *


def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


# loss
def loss_function(real, pred):
    # 去除padding，去噪声
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    # 把目标数据切分成decoder input和decoder output
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    # 获取mask
    encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(inp, tar_inp)

    # 计算梯度
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, encoder_padding_mask, decoder_mask,
                                     encoder_decoder_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    # 累积loss 和 accuracy
    train_loss(loss)
    train_accuracy(tar_real, predictions)


if __name__ == '__main__':
    # solve_cudnn_error()
    # 如果检查点存在，则恢复最新的检查点
    print(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    # 训练：遍历数据集
    # 方便可视训练过程，不是真正的NMT的训练准确度
    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    for epoch in range(epochs):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        # 训练
        for (batch, (inp, tar)) in enumerate(train_dataset.take(steps_per_epoch)):
            train_step(inp, tar)
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        # 保存
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        # 打印日志
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
            epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
