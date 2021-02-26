import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS

df = pd.read_csv('firstpart_data.csv', delimiter=',')
title = df['title'].values
body = df['body'].values
tags = df['tags'].values


def clean_data(text_arr):
    size = text_arr.shape[0]
    ps = PorterStemmer()
    clean = []
    for i in range(0, size):
        c_text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text_arr[i])
        cleanr = re.compile('<.*?>') # removing html tags from text
        cleantext = re.sub(cleanr, '', c_text)
        # cleantext = re.sub(pattern='<.*?>|', repl='', string=cleantext) # removing '|' (mai ales la tags)
        cleantext = cleantext.lower()
        cleantext_words = cleantext.split() # separare cuvinte
        cleantext_words = [ps.stem(word) for word in cleantext_words if word not in STOP_WORDS] # stem + filtrare stop_words
        cleantext = ' '.join(cleantext_words) # unire cuvinte
        clean.append(cleantext) # adaugare la lista finala

    return clean


tags_c = clean_data(tags)
print(tags_c[0:2])