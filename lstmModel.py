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

from sklearn import metrics

#os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin")

ma = pymorphy2.MorphAnalyzer()


def processingDoc(X):
    documents = []

    morph = pymorphy2.MorphAnalyzer()

    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # remove all single characters
        document = re.sub(r'\s+[а-яА-Я]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[а-яА-Я]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [morph.parse(word)[0].normal_form for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents


class Model:
    def __init__(self, max_features, maxSequenceLength):
        model_checkpoint = Sequential()
        model_checkpoint.add(Embedding(max_features, maxSequenceLength))
        model_checkpoint.add(LSTM(64, dropout=0.2, recurrent_dropout=0.0))
        model_checkpoint.add(Dense(2, activation='sigmoid'))

        model_checkpoint.compile(loss='binary_crossentropy',
                                 optimizer='adam',
                                 metrics=[tf.keras.metrics.AUC(from_logits=True)])

        print(model_checkpoint.summary())
        self.model = model_checkpoint

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

    def train(self, X_tr, y_tr, X_ts, y_ts):
        checkpoint_filepath_ = 'tmp/'
        model_checkpoint_callback_ = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath_,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        batch_size = 32
        epochs = 200
        v = 0.01
        self.model.fit(X_tr, y_tr,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(X_ts, y_ts),
                       class_weight={0: v, 1: 1 - v},
                       callbacks=model_checkpoint_callback_
                       )


def load_data_from_arrays(strings, labels, train_test_split=0.9):
    data_size = len(strings)
    test_size = int(data_size - round(data_size * train_test_split))
    print("Test size: {}".format(test_size))

    print("\nTraining set:")
    x_train = strings[test_size:]
    print("\t - x_train: {}".format(len(x_train)))
    y_train = labels[test_size:]
    print("\t - y_train: {}".format(len(y_train)))

    print("\nTesting set:")
    x_test = strings[:test_size]
    print("\t - x_test: {}".format(len(x_test)))
    y_test = labels[:test_size]
    print("\t - y_test: {}".format(len(y_test)))

    return x_train, y_train, x_test, y_test


def data_to_train(path, data_name, target_name):
    train = data_name
    target = target_name
    val = pandas.read_csv(path)
    val.dropna(inplace=True,how='any')
    val = val.reset_index(drop=True)
    val.data = val.data.astype(str)
    val.target = val.target.astype(int)

    val[train] = processingDoc(val[train])
    df = val

    descriptions = df[train]
    categories = df[target]

    # создаем единый словарь (слово -> число) для преобразования
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(descriptions.tolist())

    # Преобразуем все описания в числовые последовательности, заменяя слова на числа по словарю.
    textSequences = tokenizer.texts_to_sequences(descriptions.tolist())

    # X_train, y_train, X_test, y_test = load_data_from_arrays(descriptions, categories, train_test_split=0.8)

    X_train_, X_test_, y_train_, y_test_ = train_test_split(descriptions, categories, test_size=0.2, random_state=0)



    k = y_test_
    # Максимальное количество слов в самом длинном описании заявки
    max_words = 0
    for desc in descriptions.tolist():
        words = len(desc.split())
        if words > max_words:
            max_words = words
    print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))

    total_unique_words = len(tokenizer.word_counts)
    print('Всего уникальных слов в словаре: {}'.format(total_unique_words))

    maxSequenceLength_ = max_words

    encoder = LabelEncoder()
    encoder.fit(y_train_)
    y_train_ = encoder.transform(y_train_)
    y_test_ = encoder.transform(y_test_)

    num_classes_ = 2
    print('Количество категорий для классификации: {}'.format(num_classes_))
    vocab_size_ = round(total_unique_words / 10 )

    print(u'Преобразуем описания заявок в векторы чисел...')

    print(u'Преобразуем описания заявок в векторы чисел...')
    tokenizer = Tokenizer(num_words=vocab_size_)
    tokenizer.fit_on_texts(descriptions)

    saveTokens(tokenizer)

    X_train_ = tokenizer.texts_to_sequences(X_train_)
    X_test_ = tokenizer.texts_to_sequences(X_test_)

    X_train_ = pad_sequences(X_train_, maxlen=maxSequenceLength_)
    X_test_ = pad_sequences(X_test_, maxlen=maxSequenceLength_)

    print('Размерность X_train:', X_train_.shape)
    print('Размерность X_test:', X_test_.shape)

    print(u'Преобразуем категории в матрицу двоичных чисел '
          u'(для использования categorical_crossentropy)')
    y_train_ = keras.utils.to_categorical(y_train_, num_classes_)
    y_test_ = keras.utils.to_categorical(y_test_, num_classes_)
    print('y_train shape:', y_train_.shape)
    print('y_test shape:', y_test_.shape)

    return X_train_, X_test_, y_train_, y_test_, num_classes_, maxSequenceLength_, vocab_size_, k

def train_model():
    X_train, X_test, y_train, y_test, num_classes, maxSequenceLength, vocab_size, k = data_to_train("all_messages.csv",
                                                                                                    "data", "target")

    # максимальное количество слов для анализа
    max_features = vocab_size
    os.add_dll_directory("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin")
    print(max_features, maxSequenceLength)
    print(u'Собираем модель...')
    model = Sequential()

    save_dict({'max_features': max_features, 'maxSequenceLength': maxSequenceLength})

    model.add(Embedding(max_features, maxSequenceLength))
    model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.0))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[tf.keras.metrics.AUC(from_logits=True)])

    print(model.summary())

    batch_size = 16
    epochs = 100

    v = 0.002
    print(u'Тренируем модель...')

    checkpoint_filepath = 'tmp/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_auc',
        mode='max',
        save_best_only=True)

    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        class_weight={0: v, 1: 1 - v},
                        callbacks=model_checkpoint_callback
                        )
    model.load_weights(checkpoint_filepath)

    score = model.evaluate(X_test, y_test,
                           batch_size=batch_size, verbose=1)
    print()
    print(u'Оценка теста: {}'.format(score[0]))
    print(u'Оценка точности модели: {}'.format(score[1]))

    # Посмотрим на эффективность обучения

    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    plt.title("Эффективность обучения")
    plt.xlabel("Повторения #")
    plt.ylabel("Ошибки")
    plt.legend(loc="lower left")
    plt.show()

    y_pr = model.predict(X_test)
    print(y_pr)
    preds = [int(i[0] < i[1]) for i in y_pr]
    print(preds)
    print(list(k))
    model.save("model.h50")


def saveTokens(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def loadTokens():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer


def processing_data(data):
    tokenizer = loadTokens()
    maxSequenceLength_ = load_dict()['maxSequenceLength']
    data = tokenizer.texts_to_sequences(processingDoc([data]))
    data = pad_sequences(data, maxlen=maxSequenceLength_)
    return data


def save_dict(dictionary):
    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


def load_dict():
    with open('saved_dictionary.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict


#train_model()

model = load_model("model.h11")

def predict(data):
    data = processing_data(data)
    vl = model.predict(data)
    return int(vl[0][0] < vl[0][1])



