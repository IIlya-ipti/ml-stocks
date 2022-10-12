import re
import pandas
import pickle
import joblib
from keras.preprocessing import sequence
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
from sklearn.pipeline import Pipeline
# pipeline позволяет объединить в один блок трансформер и модель, что упрощает написание кода и улучшает его читаемость
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer преобразует тексты в числовые вектора, отражающие важность использования каждого слова из некоторого набора слов (количество слов набора определяет размерность вектора) в каждом тексте
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
# линейный классификатор и классификатор методом ближайших соседей
from sklearn import metrics
# набор метрик для оценки качества модели
from sklearn.model_selection import GridSearchCV

import pymorphy2
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

morph = pymorphy2.MorphAnalyzer()


def processingDoc(X):
    documents = []
    morph = pymorphy2.MorphAnalyzer()
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

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


def start():
    vals = pandas.read_csv("all_messages.csv")

    texts = processingDoc(vals['data'])

    vectorizer = CountVectorizer(max_features=1500, min_df=1, max_df=0.8, stop_words=stopwords.words('russian'))
    X = vectorizer.fit_transform(texts).toarray()

    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    y = [int(i) for i in list(vals['target'])]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print(y_test)

    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def testMode(model, X_train, X_test, y_train, y_test):
    classifier = model
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(list(y_pred))
    print(y_test)


def tst(model, X_train, X_test, y_train, y_test):
    predicted_sgd = model.predict(X_test)
    # print(metrics.classification_report(predicted_sgd, y_test))
    print(confusion_matrix(y_test, predicted_sgd))
    print(classification_report(y_test, predicted_sgd))
    print(accuracy_score(y_test, predicted_sgd))
    print(predicted_sgd)


def St(model, name):
    X_train, X_test, y_train, y_test = train_test_split(texts, vals['target'], test_size=0.2, random_state=0)

    sgd_ppl_clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('sgd_clf', model)])
    sgd_ppl_clf.fit(X_train, y_train)
    predicted_sgd = sgd_ppl_clf.predict(X_test)
    print(metrics.classification_report(predicted_sgd, y_test))
    print(confusion_matrix(y_test, predicted_sgd))
    print(classification_report(y_test, predicted_sgd))
    print(accuracy_score(y_test, predicted_sgd))
    print(predicted_sgd)
    save_model(sgd_ppl_clf, name)


def save_model(clf, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(clf, f)


def load_model(name):
    # load
    with open(name + '.pkl', 'rb') as f:
        clf2 = pickle.load(f)
        return clf2


def first():
    class_weight = {0: 0.01, 1: 1 - 0.01}

    svc = SVC(kernel='sigmoid', gamma=1.0, class_weight=class_weight)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.6)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111, class_weight=class_weight)
    lrc = LogisticRegressionCV(solver='liblinear', penalty='l1', class_weight=class_weight)
    rfc = RandomForestClassifier(n_estimators=62, random_state=111, class_weight=class_weight)
    abc = AdaBoostClassifier(n_estimators=62, random_state=111)
    bc = BaggingClassifier(n_estimators=14, random_state=111)
    etc = ExtraTreesClassifier(n_estimators=10, random_state=111, class_weight=class_weight)

    print("svc")
    testMode(svc, X_train, X_test, y_train, y_test)
    print("knc")
    testMode(knc, X_train, X_test, y_train, y_test)
    print("mnb")
    testMode(mnb, X_train, X_test, y_train, y_test)
    print("dtc")
    testMode(dtc, X_train, X_test, y_train, y_test)
    print("lrc")
    testMode(lrc, X_train, X_test, y_train, y_test)
    print("abc")
    testMode(abc, X_train, X_test, y_train, y_test)
    print("bc")
    testMode(bc, X_train, X_test, y_train, y_test)
    print("rfc")
    testMode(rfc, X_train, X_test, y_train, y_test)
    print("etc")
    testMode(etc, X_train, X_test, y_train, y_test)


print("new")


def second():
    class_weight = {0: 0.01, 1: 1 - 0.01}

    svc = SVC(kernel='sigmoid', gamma=1.0, class_weight=class_weight)
    knc = KNeighborsClassifier(n_neighbors=49)
    mnb = MultinomialNB(alpha=0.6)
    dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111, class_weight=class_weight)
    lrc = LogisticRegressionCV(solver='liblinear', penalty='l1', class_weight=class_weight)
    rfc = RandomForestClassifier(n_estimators=50, random_state=111, class_weight=class_weight)
    abc = AdaBoostClassifier(n_estimators=62, random_state=111)
    bc = BaggingClassifier(n_estimators=14, random_state=111)
    etc = ExtraTreesClassifier(n_estimators=10, random_state=111, class_weight=class_weight)
    print("svc")
    St(svc, "svc")
    print("knc")
    St(knc, "knc")
    print("mnb")
    St(mnb, "mnb")
    print("dtc")
    St(dtc, "dtc")
    print("lrc")
    St(lrc, "lrc")
    print("rfc")
    St(rfc, "rfc")
    print("abc")
    St(abc, "abc")
    print("bc")
    St(bc, "bc")
    print("etc")
    St(etc, "etc")


# X_train, X_test, y_train, y_test = train_test_split(texts, vals['target'], test_size=0.2, random_state=0)
#second()

# md = load_model("rfc")


def predict(data):
    return md.predict(processingDoc([data]))
