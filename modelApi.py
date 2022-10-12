import pickle

import keras
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils.data_utils import pad_sequences

import re
import pymorphy2


# this method for processing
def processing_doc(X):
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

def save_dict(dictionary, name):
    with open(name + 'dict' + '.pkl', 'wb') as f:
        pickle.dump(dictionary, f)


def load_dict(name):
    with open(name + 'dict' + '.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        return loaded_dict


def save_token(tokenizer, name):
    with open("tokens/" + name + '.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokens(name):
    with open("tokens/" + name + '.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer


def processing_data(train):
    # clean train data
    return processing_doc(train)


def new_target_encoder(train_data):
    encoder = LabelEncoder()
    encoder.fit(train_data)
    return encoder, encoder.classes_


def max_word_size(data):
    # Максимальное количество слов в самом длинном описании заявки
    max_words = 0
    for desc in data.tolist():
        words = len(desc.split())
        if words > max_words:
            max_words = words

    print('Максимальное количество слов в самом длинном описании заявки: {} слов'.format(max_words))
    return max_words


def total_unique_words(data):
    # создаем единый словарь (слово -> число) для преобразования
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data.tolist())
    return len(tokenizer.word_counts)


class ModelClassifier:
    def __init__(self, model=None, name=None, load=False):
        self.train_data = None
        self.train_target_data = None
        self.test_data = None
        self.test_target_data = None
        self.name = name
        if load:
            self.model = keras.models.load_model(name)
            self.tokenizer = self.__load_tokenizer()
            dct = load_dict(self.name)
            self.max_words_in_text_size = dct['max_words_in_text_size']
            self.after_processing_size = dct['after_processing_size']
        else:
            self.model = model
            self.after_processing_size = None
            self.max_words_in_text_size = None
            self.tokenizer = None
        self.number_of_classes = None
        self.target_encoder = None

    def __save_tokenizer(self, tokenizer):
        save_token(tokenizer, self.name)

    def __load_tokenizer(self):
        return load_tokens(self.name)

    def __target_encode(self, data):
        return self.target_encoder.transform(data)

    def __set_new_tokenizer(self, data):
        self.after_processing_size = round(total_unique_words(data) / 10)

        # create new tokenizer (word -> number)
        tokenizer = Tokenizer(num_words=self.after_processing_size)
        tokenizer.fit_on_texts(data)
        return tokenizer

    def __tokenize_words(self, data):
        data = self.tokenizer.texts_to_sequences(data)
        data = pad_sequences(data, maxlen=self.max_words_in_text_size)
        return data

    def __set_training_data(self, data, data_name, target_name):
        # clean data of nan
        data.dropna(inplace=True, how='any')
        data.reset_index(drop=True, inplace=True)

        ##
        data.data = data.data.astype(str)
        data.target = data.target.astype(int)

        ##
        data[data_name] = processing_data(data[data_name])

        # set train data and target data
        X_train, X_test, y_train, y_test = train_test_split(data[data_name], data[target_name], test_size=0.2,
                                                            random_state=0)
        self.train_target_data = y_train
        self.test_target_data = y_test
        self.target_encoder, self.max_words_in_text_size = new_target_encoder(y_train)
        self.max_words_in_text_size = max_word_size(X_train)
        self.tokenizer = self.__set_new_tokenizer(X_train)
        self.train_data = self.__tokenize_words(X_train)
        self.test_data = self.__tokenize_words(X_test)
        self.model = self.model(self.after_processing_size, self.max_words_in_text_size)

        # for binary entropy
        self.train_target_data = keras.utils.to_categorical(self.__target_encode(y_train), self.number_of_classes)
        self.test_target_data = keras.utils.to_categorical(self.__target_encode(y_test), self.number_of_classes)

        # save params
        save_dict({'after_processing_size': self.after_processing_size,
                   'max_words_in_text_size': self.max_words_in_text_size}, self.name)
        save_token(self.tokenizer, name=self.name)

    def train(self, data, data_name, target_name):
        """
        this function for train Model
        :param target_name: name of target column
        :param data_name: name of data column
        :param data: pandas array
        """
        self.__set_training_data(data, data_name, target_name)
        self.model.train(
            self.train_data,
            self.train_target_data,
            self.test_data,
            self.test_target_data
        )
        self.model.save_model(self.name)

    def predict(self, data):
        data = processing_data([data])
        data = self.__tokenize_words(data)
        pred = self.model.predict(data)
        return int(pred[0][0] < pred[0][1])
