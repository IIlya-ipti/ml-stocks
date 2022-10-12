import re
import pymorphy2


# this method for processing 
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

