
import keras
import pandas
import pymorphy2
import re
import keras.metrics
from keras.utils.data_utils import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from sklearn import metrics
import tensorflow as tf
from tensorflow import keras

from keras import layers

import lstmModel


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


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
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


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def data_to_train(path, data_name, target_name):
    train = data_name
    target = target_name
    val = pandas.read_csv(path)
    val.dropna(inplace=True, how='any')
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
    vocab_size_ = round(total_unique_words / 1)

    print(u'Преобразуем описания заявок в векторы чисел...')

    print(u'Преобразуем описания заявок в векторы чисел...')
    tokenizer = Tokenizer(num_words=vocab_size_)
    tokenizer.fit_on_texts(descriptions)

    lstmModel.saveTokens(tokenizer)

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


x_train, x_val, y_train, y_val, num_classes, maxSequenceLength, vocab_size, k = data_to_train("all_messages.csv",
                                                                                              "data", "target")

# Only consider the top 20k words
maxlen = maxSequenceLength  # Only consider the first 200 words of each movie review

lstmModel.save_dict({'max_features': vocab_size, 'maxSequenceLength': maxSequenceLength})

print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
#x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
#x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab _size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

batch_size = 32
v = 0.02

'' 'Модели обучение' '' ''
model.compile("adam", "binary_crossentropy", metrics=[tf.keras.metrics.AUC(from_logits=True)])

checkpoint_filepath = 'tmp/'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_auc',
    mode='max',
    save_best_only=True)

history = model.fit(
    x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_val, y_val),
    class_weight={0: v, 1: 1 - v},callbacks=model_checkpoint_callback
)
model.load_weights(checkpoint_filepath)
score = model.evaluate(x_val, y_val,
                           batch_size=batch_size, verbose=1)
print()
print(u'Оценка теста: {}'.format(score[0]))
print(u'Оценка точности модели: {}'.format(score[1]))

y_pr = model.predict(x_val)
print(y_pr)
preds = [int(i[0] < i[1]) for i in y_pr]
print(preds)
print(list(k))

model.save("model.h12")
