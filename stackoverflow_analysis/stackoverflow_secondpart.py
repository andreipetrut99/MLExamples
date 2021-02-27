from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('firstpart_data.csv', delimiter=',')
tags = df['tags'].values

all_tags = []
for tag in tags:    # despartire taguri dupa |
    all_tags += tag.split('|')

counter = Counter(all_tags) # dictionar cu frecventele
# vreau sa iau cele mai mari valori deci trebuie sa transform intr-un tip de data care poate fi sortat

tags = pd.DataFrame() # df cu frecventele
tags.transpose()
tags['tag'] = list(counter.keys())
tags['count'] = list(counter.values())
tags = tags.sort_values(by='count', ascending=False)   # sortare


plt.barh(list(tags['tag'])[:10], list(tags['count'])[:10]) # selectez doar primele 10 taguri pentru plt
plt.ylabel('Tags')
plt.xlabel('Freq')
plt.title('Top 10 most freq tags')
plt.show()


tags_no = []
all_tags = list(df['tags'].unique())
for tag in all_tags:
    tags_no.append(len(tag.split('|'))) # adaugare nr de taguri

tags_no = Counter(tags_no)

plt.bar(list(tags_no.keys()), list(tags_no.values()))
plt.xlabel('Numar tag-uri')
plt.ylabel('Numar intrebari')
plt.title('Numar intrebari / Numar taguri')
plt.show()




