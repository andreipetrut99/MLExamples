import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def get_data(filename):
    df = pd.read_csv(filename, delimiter=',')
    labels = df['type'].values
    messages = df['text'].values

    pos_examples = np.sum(labels == 'ham') / labels.shape[0]
    neg_examples = 1 - pos_examples

    shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in shuffle_stratified.split(messages, labels):
        msg_train, msg_test = messages[train_index], messages[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

    # y_train = np.int32(labels_train == 'ham')
    # y_test = np.int32(labels_test == 'spam')

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(labels_train)

    y_train = label_binarizer.transform(labels_train)
    y_test = label_binarizer.transform(labels_test)

    return msg_train, msg_test, y_train, y_test


msg_train, msg_test, y_train, y_test = get_data('sms_spam.csv')

count_vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')
count_vectorizer.fit(msg_train)
X_train = count_vectorizer.transform(msg_train)
X_test = count_vectorizer.transform(msg_test)

model = MultinomialNB(alpha=1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy score of Y test {}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions))
