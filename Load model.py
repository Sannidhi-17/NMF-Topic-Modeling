# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:50:03 2021

@author: Sannidhi
"""
#Use CountVectorizer to get bigrams to visualize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_pickle('text_preprocessing.pkl')
#print(df.head())
cv = CountVectorizer(max_df = 0.95, min_df = 2, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(df['article'])
#print(X)

# Most frequently occuring words
def get_top_n_words(corpus,n=None):
    vec = CountVectorizer().fit(df['article'])
    bag_of_words=vec.transform(df['article'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word , idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
# Convert most freq words to datafame for visuals

top_words = get_top_n_words(df['article'], n=20)
top_df = pd.DataFrame(top_words)
top_df.columns = ['Word', 'Freq']



# Term vectorization term weighting:
tfidf_vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 2,max_features=None, ngram_range=(1,1),
                                  analyzer='word')
tfidf = tfidf_vectorizer.fit_transform(df['article'])
# get the feature names
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

#print('Vocabulary has %d distinct terms' % len(tfidf_feature_names))

import operator
def rank_terms( tfidf, tfidf_feature_names ):
    # get the sums over each column
    sums = tfidf.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(tfidf_feature_names):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

ranking = rank_terms(tfidf, tfidf_feature_names)
#for i, pair in enumerate( ranking[0:20] ):
    #print( "%02d. %s (%.0f)" % ( i+1, pair[0], pair[1] ) )
    



# NMF model fitting

no_topics = 10
nmf=NMF(n_components = no_topics, random_state =1, alpha=0.1, l1_ratio = 0.5, init= 'nndsvd').fit(tfidf)

def display_topics(model, feature_names, no_top_words):
    col1 = 'topic'
    col2 = 'top_ten_words'
    dct = {col1: [], col2: []}
    for topic_idx, topic in enumerate(model.components_):
        dct[col1].append(int(topic_idx) +1)
        dct[col2].append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return pd.DataFrame.from_dict(dct)

no_top_words = 10
topic_word = display_topics(nmf, tfidf_feature_names, no_top_words)

no_top_words = 3
topic_word_3 = display_topics(nmf, tfidf_feature_names, no_top_words)
topic_word_3['Top_3_keywords'] = topic_word_3.top_ten_words.str.title()
topic_word = topic_word_3.loc[:,['topic', 'Top_3_keywords']]



nmf_W = nmf.transform(tfidf)
nmf_H = nmf.components_

df2 = pd.DataFrame({'topic':nmf_W.argmax(axis=1),
                    'documents': df['documents']},
                  columns = ['topic', 'documents'])
df2.to_excel ('topic_number_with_responses.xlsx')
df2.to_pickle('topic.pkl')



#print("----")
x = []
topic_word['documents'] = ''
for i in range(no_topics):
    df3 = df2[df2['topic']==i]     
    x1 = df3['documents'].tolist()
    column_values = pd.Series(x1)
    topic_word.iat[i, topic_word.columns.get_loc('documents')] = column_values
    i+=1
    
#df = pd.DataFrame(column_values)
#print(x1)
#print("***")
#print(df2.head())
x = df2['topic'].value_counts()
ser = pd.Series(x) 
#print(ser)

y = df2['topic'].value_counts(normalize=True) * 100
#ser1 = pd.Series(y).astype(str) + '%'
ser1 = pd.Series(y)
topic_word['frequency'] = pd.Series(ser)
#print(topic_word['frequency'].dtype)
topic_word['percentage_of_each_topic'] = pd.Series(ser1)
topic_word['percentage_of_each_topic'] = topic_word.percentage_of_each_topic.astype(float)
topic_word['percentage_of_each_topic'] = np.around(topic_word['percentage_of_each_topic'], decimals=2)




topic_word.to_pickle('theme_keywords.pkl')


# plot the bar graph per topic with top 3 keywords
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(4, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
        fig.savefig("topic wise top 3 keywords.png")
plot_top_words(nmf, tfidf_feature_names, 3,
               'Per topic top 3 keywords')


# word frequency per topic
nmf.fit(tfidf)
# Transform the TF-IDF: nmf_features
nmf_features = nmf.transform(tfidf)
#print(nmf.components_.shape)

# Create a DataFrame: components_df
components_df = pd.DataFrame(nmf.components_, columns=tfidf_feature_names)
#print(components_df.head())
  
for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    #print(f'For topic {topic+1} the words with the highest value are:')
    #print(tmp.nlargest(10))
    #print('\n')
    
    
tmp = components_df.iloc[topic]
a = tmp.nlargest(10)

####
'''
snippets = df[:100]
def get_top_snippets(all_snippets, W, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( W[:,topic_index] )[::-1]
    # now get the snippets corresponding to the top-ranked indices
    top_snippets = []
    for doc_index in top_indices[0:top]:
        top_snippets.append( all_snippets[doc_index] )
    return top_snippets

topic_snippets = get_top_snippets(snippets, nmf_features, 1, 10 )
for i, snippet in enumerate(topic_snippets):
    print("%02d. %s" % ( (i+1), snippet ) )
'''

def plot_top_term_weights( terms, H, topic_index, top ):
    # get the top terms and their weights
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    # note we reverse the ordering for the plot
    top_terms.reverse()
    top_weights.reverse()
    # create the plot
    fig = plt.figure(figsize=(13,8))
    # add the horizontal bar chart
    ypos = np.arange(top)
    ax = plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)
    plt.xlabel("Term Weight",fontsize=14)
    plt.tight_layout()
    plt.show()

plot_top_term_weights(tfidf_feature_names, nmf_H, 6, 15 )

#  top 5 articles per topic
snippets = df['documents']
def get_top_snippets(all_snippets, W, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( W[:,topic_index-1] )[::-1]
    # now get the snippets corresponding to the top-ranked indices
    top_snippets = []
    for doc_index in top_indices[0:top]:
        top_snippets.append( all_snippets[doc_index] )
    return top_snippets
'''
topic_snippets = get_top_snippets( snippets, nmf_W, 19, 5)
for i, snippet in enumerate(topic_snippets):
    print("%02d. %s" % ( (i+1), snippet ) )
'''

topic_word['xxx'] = ''
topic_snippets = None
for k in range(1,no_topics+1):
    print(k)
    topic_snippets = get_top_snippets( snippets, nmf_W, k, 5)
    z = pd.Series(topic_snippets)
    topic_word.iat[k - 1, topic_word.columns.get_loc('xxx')] = z
    
    
topic_word.to_excel ('themes_keywords.xlsx')
"""
    for j, snippet in enumerate(topic_snippets):
         y = "%02d. %s" % ((j+1), snippet )
         print(y)
         #z.to_excel('top_five.xlsx')
         j += 1
    #z = pd.Series(five_topics)
    #topic_word.iat[i, topic_word.columns.get_loc('xxx')] = z
    #print(z)
topic_word.iat[i, topic_word.columns.get_loc('xxx')] = z
topic_word.to_excel ('themes_keywords.xlsx')
#df4.to_excel('top_five.xlsx')
"""

