from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
from wordcloud import WordCloud

df = pd.read_csv('firstpart_data.csv', delimiter=',')
tags = df['tags'].values

all_tags = []
for tag in tags:    # despartire taguri dupa |
    all_tags += tag.split('|')

counter = Counter(all_tags) # dictionar cu frecventele
# vreau sa iau cele mai mari valori deci trebuie sa transform intr-un tip de data care poate fi sortat

all_tags_string = ' '.join(all_tags)

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

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_tags_string)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

ml_tags = ['python', 'r', 'excel', 'sql']
x = []
y = []
for tag in ml_tags:
    index = list(tags['tag']).index(tag)
    x.append(tag)
    y.append(list(tags['count'])[index]) # am adaugat in y frecventa tagurilor din x

plt.bar(x, y)
plt.ylabel('Tag')
plt.xlabel('Aparitii tag')
plt.title('Popularitate taguri ML')
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




