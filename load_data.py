#!/usr/bin/python3
# -*- coding:utf-8 -*-
import unicodedata
import re
import jieba
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(s):
    # 转化成ascii，变小写去空格
    s = unicode_to_ascii(s.lower().strip())
    # 标点符号前后加空格
    s = re.sub(r'([?.!,。，！？‘’“”()（）])', r' \1 ', s)
    # 多余的空格变成一个空格
    s = re.sub(r'[" "]+', ' ', s)
    # 除了标点符号和字母外都是空格
    # s = re.sub(r'[^a-zA-Z?.!,¿]', ' ', s)
    # 去掉前后空格
    s = s.rstrip().strip()
    # 前后加标记
    s = '<start> ' + s + ' <end>'
    return s


# 解析文件
def parse_data(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    sentence_pairs = [line.split('\t') for line in lines]
    preprocess_sentence_pairs = [
        (preprocess_sentence(en), preprocess_sentence(' '.join(jieba.cut_for_search(cmn)))) for en, cmn in
        sentence_pairs]
    # 解包和zip联用：将每一个元组解开，重新组合成两个新的列表
    return zip(*preprocess_sentence_pairs)


def parse_enbigdata(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    en_sentence = [preprocess_sentence(line) for line in lines]
    return en_sentence


def parse_cmnbigdata(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    cmn_sentence = [preprocess_sentence(' '.join(jieba.cut_for_search(line))) for line in lines]
    return cmn_sentence


def tokenizer(lang):
    lang_tokenizer = keras.preprocessing.text.Tokenizer(num_words=None, filters='', split=' ')
    # 统计词频，生成词表
    lang_tokenizer.fit_on_texts(lang)
    # id化
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # padding
    tensor = keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


# 求最大长度
def max_length(tensor):
    return max(len(t) for t in tensor)


# 验证tokenizer是否转化正确
def convert(example, tokenizer):
    for t in example:
        if t != 0:
            print('%d --> %s' % (t, tokenizer.index_word[t]))



def make_dataset(input_tensor, output_tensor, batch_size, epochs, shuffle, buffer_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, output_tensor))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset


# 解析文件
en_cmn_file_path = 'data/cmn_proc.txt'
en_dataset, cmn_dataset = parse_data(en_cmn_file_path)

en_file_path = 'data/news-commentary-v12.zh-en.en'
cmn_file_path = 'data/news-commentary-v12.zh-en.zh'
en_dataset_b = en_dataset + tuple(parse_enbigdata(en_file_path))
cmn_dataset_b = cmn_dataset + tuple(parse_cmnbigdata(cmn_file_path))

# tokenizer
input_tensor, input_tokenizer = tokenizer(cmn_dataset)
output_tensor, output_tokenizer = tokenizer(en_dataset)

# 求最大长度
max_length_input = max_length(input_tensor)
max_length_output = max_length(output_tensor)
print(max_length_input, max_length_output)

# 切分训练集和验证集
input_train, input_eval, output_train, output_eval = train_test_split(input_tensor, output_tensor, test_size=0.2)

# data参数
buffer_size = 10000 #20000
batch_size = 32 # 64
epochs = 100

# 构建dataset
train_dataset = make_dataset(input_train, output_train, batch_size, epochs, True, buffer_size)
eval_dataset = make_dataset(input_eval, output_eval, batch_size, 1, False, buffer_size)

for x, y in train_dataset.take(1):
    print(x.shape, y.shape)
