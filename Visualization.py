# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:19:27 2021

@author: Sannidhi
"""

'''
steps:
    1. frequency of each topic graph
    2. Percentage of each topic graph
    
'''
import pandas as pd
import matplotlib.pyplot 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_pickle('E:/Internship_finland/NMF/Validation/theme_keywords.pkl')
print(df.head())

# Frequency plot
plt.figure(figsize=(20, 10))
splot=sns.barplot(x="topic",y="frequency",data=df, color= 'blue')
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.1f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 20), 
                   textcoords = 'offset points')
plt.xlabel("Topics", size=18)
plt.ylabel("Frequency", size=18)
plt.savefig("Frequency of topic.png")


plt.figure(figsize=(20, 10))

splot=sns.barplot(x="topic", y="percentage_of_each_topic",data = df, color= 'blue')

for p in splot.patches:
    splot.annotate('{:.1f}%'.format(p.get_height()), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 20), 
                   textcoords = 'offset points')

#without_hue(splot, df.percentage_of_each_topic)    
plt.xlabel("Topics", size=18)
plt.ylabel("percentage", size=18)
plt.savefig("percentage of topic.png")