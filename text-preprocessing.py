# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:02:19 2021

@author: Sannidhi
"""
'''
Steps for this file:
    1. Import libraries
    2. import data file
    3. Lower case
    4. tokens
    5. lemmatize
    6. sremove stop words
    7. string conversion
    8. save as a excel and pkl format
    9. word cloud of lemmatized and after removed stop words 
'''
    


import pandas as pd
# Word Cloud
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from nltk.tokenize import RegexpTokenizer
import libvoikko
from nltk.corpus import stopwords
import tabula
import os


file_name = input("Enter Filename: ")


if file_name.endswith('.txt'):
    f = open(file_name,  encoding="utf8")   
    read_file = pd.read_csv (f, header=None, error_bad_lines=False)
    x = os.path.splitext(file_name)[0]
    fname = x + '.csv'
    df = read_file.to_csv (fname, index=None)
    print('file is txt')
    
elif file_name.endswith('.xlsx'):
   #f = open(file_name, encoding= 'unicode_escape')
   read_file = pd.read_excel(file_name)
   x = os.path.splitext(file_name)[0]
   fname = x + '.csv'
   df = read_file.to_csv (fname, index = None, header=True)
   print('file is excel')
else:
    print('none of them')
    x = os.path.splitext(file_name)[0]
    fname = x + '.csv'
    df = tabula.convert_into(fname, "file_name.csv")


df = pd.read_csv(fname, encoding='ISO-8859-1',
                        sep=',', error_bad_lines=False, names=['documents'])
print(df.head())

#df = pd.read_csv("E:/Internship_finland/NMF/Validation/NMF topic modelling case corpus 120221.csv", encoding='ISO-8859-1',
 #                       sep=',', error_bad_lines=False, names=['documents'])


df['index'] = np.arange(1, len(df)+1)
df = df[['index', 'documents']]

# Converting into lower case
df['documents'] = df.documents.map(lambda x: x.lower())


# Convert articles ino the tokens

df['docuemnt_tokens'] = df.documents.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

# Apply Lemmatization (Voikko)

C = libvoikko.Voikko(u"fi")

def lemmatize_text(text):
    bf_list = []
    for w in text:
        voikko_dict = C.analyze(w)
        if voikko_dict:
            bf_word = voikko_dict[0]['BASEFORM']
        else:
            bf_word = w
        bf_list.append(bf_word)
    return bf_list

df['lemmatized'] = df.docuemnt_tokens.apply(lemmatize_text)


stop_en = stopwords.words('finnish')
df['article'] = df.lemmatized.map(lambda x: [t for t in x if t not in stop_en]) 


# make sure the datatype of column 'article_removed_stop_words' is string
df['article'] = df['article'].astype(str)
df['article'] = df['article'].apply(eval).apply(' '.join)
df.to_excel("text_preprocessing.xlsx")
df.to_pickle('text_preprocessing.pkl')
print(df.head())

#wordcloud with black background

text = df['article'].values 
wordcloud = WordCloud(width=1600, height=800, background_color= 'black').generate(" ".join(text))
# Open a plot of the generated image.

plt.figure(figsize=(20,10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('wordcloud_lemmatized word.png', facecolor='k', bbox_inches='tight')
