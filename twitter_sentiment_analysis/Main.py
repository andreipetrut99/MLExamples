import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def getdata():
    df = pd.read_csv('train.csv', delimiter=',')
    labels = df['label'].values
    tweets = df['tweet'].values

    print(tweets[160])
    # positive / negative examples
    pos_examples = np.sum(labels == 0) / labels.shape[0]
    neg_examples = 1 - pos_examples
    print("Positive examples of tweets= {}".format(pos_examples))
    print("Negative examples of tweets= {}".format(neg_examples))

    # debalanced dataset => random split wouldn't be effective
    shuffle_stratified = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train_index, test_index in shuffle_stratified.split(tweets, labels):
        tweets_train, tweets_test = tweets[train_index], tweets[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

    return tweets_train, labels_train, tweets_test, labels_test


tweets_train, y_train, tweets_test, y_test = getdata()


# # reduce len of words and remove '@user'
# tw_tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
# tokens_train = [tw_tokenizer.tokenize(tweet) for tweet in tweets_train]
# tokens_test = [tw_tokenizer.tokenize(tweet) for tweet in tweets_test]
#
# vectorizer = CountVectorizer(lowercase=True, analyzer='word', stop_words='english')
# ts = vectorizer.fit_transform(tweets_test).toarray()
# print(vectorizer.vocabulary_)
#

def clean_data(tweets):
    clean_tweets = []  # final tweets
    size = tweets.shape[0]
    ps = PorterStemmer()
    # taking every tweet and 'clean' it
    for i in range(0, size):
        tweet = re.sub(pattern='[^a-zA-Z]', repl=' ', string=tweets[i])  # taking each tweet
        tweet = re.sub(pattern='user', repl='', string=tweet)  # removing its @user
        tweet = tweet.lower()  # lowercase
        tweet_words = tweet.split()  # split tweet and get its words
        tweet_words = [ps.stem(word) for word in tweet_words if
                       word not in STOP_WORDS]  # stem each word and filter stopwords
        tweet = ' '.join(tweet_words)  # build the tweet back again
        clean_tweets.append(tweet)
    return clean_tweets


clean_train_tweets = clean_data(tweets_train)
clean_test_tweets = clean_data(tweets_test)

size = len(clean_train_tweets)
normal_words = []
racist_words = []
for i in range(0, size):
    if y_train[i] == 0:
        normal_words.append(clean_train_tweets[i])
    else:
        racist_words.append(clean_train_tweets[i])

# normal_words = ' '.join([text for text in normal_words])
# racist_words = ' '.join([text for text in racist_words])
# wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(racist_words)
# plt.figure(figsize=(10, 7))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis('off')
# plt.show()

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(clean_train_tweets)
X_test = vectorizer.transform(clean_test_tweets)

model = MultinomialNB(alpha=0.2)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy score of test {}".format(accuracy_score(y_test, predictions)))
print(classification_report(y_test, predictions))
