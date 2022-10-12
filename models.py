import re
import os

import pickle
import tensorflow as tf
from keras.models import load_model
import keras
import pandas
from keras.preprocessing import sequence
# Посмотрим на эффективность обучения
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, ConvLSTM2D
import pymorphy2
import re
import keras.metrics
from keras.utils.data_utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import nltk
from pandas._libs.tslibs.offsets import Second
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from keras import layers

from sklearn import metrics


# os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin")

class Model:
    def __init__(self, model):
        self.model = model

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def train(self, X_tr, y_tr, X_ts, y_ts):
        checkpoint_filepath = 'tmp/'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
        batch_size = 32
        v = 0.02
        epochs = 1
        self.model.fit(
            X_tr, y_tr, batch_size=batch_size, epochs=epochs, validation_data=(X_ts, y_ts),
            class_weight={0: v, 1: 1 - v}, callbacks=model_checkpoint_callback
        )
        self.model.load_weights(checkpoint_filepath)
        score = self.model.evaluate(X_ts, y_ts,
                                    batch_size=batch_size, verbose=1)
        print()
        print(u'Оценка теста: {}'.format(score[0]))
        print(u'Оценка точности модели: {}'.format(score[1]))

    def save_model(self, path):
        self.model.save("models/" + path)


class LstmModel(Model):
    def __init__(self, max_features, maxSequenceLength):
        model_checkpoint = Sequential()
        model_checkpoint.add(Embedding(max_features, maxSequenceLength))
        model_checkpoint.add(LSTM(64, dropout=0.2, recurrent_dropout=0.0))
        model_checkpoint.add(Dense(2, activation='sigmoid'))

        model_checkpoint.compile(loss='binary_crossentropy',
                                 optimizer='adam',
                                 metrics=[tf.keras.metrics.AUC(from_logits=True)])
        super().__init__(model_checkpoint)


class _MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(_MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class _TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(_TransformerBlock, self).__init__()
        self.att = _MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class _TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(_TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class TransformerModel(Model):
    def __init__(self, max_features, maxSequenceLength):

        embed_dim = 32  # Embedding size for each token
        num_heads = 4  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        inputs = layers.Input(shape=(maxSequenceLength,))
        embedding_layer = _TokenAndPositionEmbedding(maxSequenceLength, max_features, embed_dim)
        x = embedding_layer(inputs)
        transformer_block = _TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        outputs = layers.Dense(2, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(from_logits=True)])
        super().__init__(model)


