#!/usr/bin/python3
# -*- coding:utf-8 -*-
import tensorflow as tf
import jieba
from utils import *
from hyperparameter import input_tokenizer, output_tokenizer, max_length, transformer, ckpt, ckpt_manager
from load_data import preprocess_sentence

def evaluate(inp_sentence):
    # 预处理
    inp_sentence = preprocess_sentence(' '.join(jieba.cut_for_search(inp_sentence)))
    # text到id转化
    inputs = [input_tokenizer.word_index[token] for token in inp_sentence.split(' ')]
    # padding
    # inputs = keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length, padding='post')
    # 转化为tensor
    inputs = tf.convert_to_tensor(inputs)

    # 扩维
    # encoder_input.shape: (1, input_seq_len)
    encoder_input = tf.expand_dims(inputs, 0)
    # decoder_input.shape: (1, 1)
    decoder_input = tf.expand_dims([output_tokenizer.word_index['<start>']], 0)

    for i in range(max_length):
        # 创建mask
        encoder_padding_mask, decoder_mask, encoder_decoder_padding_mask = create_masks(
            encoder_input, decoder_input)

        # 预测
        # predictions.shape: (batch_size, output_target_len, target_vocab_size)
        predictions, attention_weights = transformer(
            encoder_input, decoder_input, False, encoder_padding_mask,
            decoder_mask, encoder_decoder_padding_mask)
        # 取出预测序列的最后一个
        # predictions.shape: (batch_size, target_vocab_size)
        predictions = predictions[:, -1, :]
        # 取最大值
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # 判断是否是最后一位
        if tf.equal(predicted_id, output_tokenizer.word_index['<end>']):
            # 因为之前扩维，故把第0个维度缩减
            return tf.squeeze(decoder_input, axis=0), attention_weights

        decoder_input = tf.concat([decoder_input, [predicted_id]], axis=-1)
    return tf.squeeze(decoder_input, axis=0), attention_weights


def plot_encoder_decoder_attention(attention, input_sentence, result, layer_name):
    """可视化Encoder和Decoder之间的attention_weights"""
    fig = plt.figure(figsize=(16, 8))
    # input_id_sentence = input_tokenizer.encode(input_sentence)
    # 预处理
    inp_sentence = preprocess_sentence(' '.join(jieba.cut_for_search(input_sentence)))
    # text到id转化
    inputs = [input_tokenizer.word_index[token] for token in inp_sentence.split(' ')]

    # attention[layer_name].shape: (batc_size=1, num_heads, tar_len, input_len)
    # attention.shape: (num_heads, tar_len, input_len)
    attention = tf.squeeze(attention[layer_name], axis=0)

    # 画num_heads个子图
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)
        # 画矩阵,去掉最后一个
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        # 设置字体
        fontdict = {'fontsize': 10}

        # 设置锚点
        ax.set_xticks(range(len(inputs)))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result) - 1.5, -0.5)

        # 设置label
        ax.set_xticklabels([input_tokenizer.index_word[i] for i in inputs], fontdict=fontdict,
            rotation=90)
        ax.set_yticklabels(result, fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))

    # 自适应调整子图位置、间距
    plt.tight_layout()
    plt.show()


def translate(input_sentence, layer_name=''):
    result, attention_weights = evaluate(input_sentence)
    predicted_sentence = []
    for i in result.numpy():
        word = output_tokenizer.index_word[i]
        predicted_sentence.append(word)
    predicted_sentence.append('<end>')
    predicted_sentences = ' '.join(predicted_sentence)

    print('Input: {}'.format(input_sentence))
    print('Predicted translation: {}'.format(predicted_sentences))

    if layer_name:
        plot_encoder_decoder_attention(attention_weights, input_sentence, predicted_sentence, layer_name)


if __name__ == "__main__":
    # 如果检查点存在，则恢复最新的检查点
    print(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!')

    translate('啊。')
    translate('你在干什么？')
    translate('你就随了我的意吧。')
    translate('我父母通常用法语对话，即使我母亲的母语是英语。')
    translate('你在家吗？')
    translate('汤姆正在教我怎么开帆船。', layer_name='decoder_layer4_att2')
