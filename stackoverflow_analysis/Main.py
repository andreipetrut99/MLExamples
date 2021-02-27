import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('firstpart_data.csv', delimiter=',')
title = df['title'].values
body = df['body'].values
tags = df['tags'].values


def clean_data(text_arr):
    size = text_arr.shape[0]
    ps = PorterStemmer()
    clean = []
    for i in range(0, size):
        cleanr = re.compile('<.*?>') # removing html tags from text
        cleantext = re.sub(cleanr, '', str(text_arr[i]))
        cleantext = re.sub(pattern='\|', repl=' ', string=cleantext) # removing '|' (mai ales la tags)
        cleantext = re.sub(pattern=r'[0-9]+', repl='', string=cleantext)
        # cleantext = str(text_arr[i])
        cleantext = cleantext.lower()
        cleantext_words = cleantext.split() # separare cuvinte
        cleantext_words = [ps.stem(word) for word in cleantext_words if word not in STOP_WORDS] # stem + filtrare stop_words
        cleantext = ' '.join(cleantext_words) # unire cuvinte
        clean.append(cleantext) # adaugare la lista finala

    return clean


tags_c = clean_data(tags)
body_c = clean_data(body)
title_c = clean_data(title)

final_strings = [tags_c[i]
                 + ' ' + tags_c[i]
                 + ' ' + body_c[i]
                 + ' ' + title_c[i] for i in range(0, len(tags_c))]

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer.fit(final_strings)
# matrix = tfidf_vectorizer.transform(final_strings).toarray()
#
# first_doc_scores = matrix[0]
#
# df_tfidf = pd.DataFrame(first_doc_scores, index=tfidf_vectorizer.get_feature_names(), columns=['tf_idf_scores'])
# df_tfidf.sort_values(by=['tf_idf_scores'])
# print(df_tfidf)

count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))
for i in range(0, len(final_strings)):
    matrix = count_vectorizer.fit_transform(final_strings[i])
    df_c = pd.DataFrame(matrix.sum(axis=0).T,
                        index=count_vectorizer.get_feature_names(),
                        columns=['freq'].sort_values(by='freq', ascending=False).plot(kind='bar', title='TITLe'))
