# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:07:04 2021

@author: Sannidhi
"""
import os
import tabula
import pandas as pd
import numpy as np

file_name = input("Enter a file name: ")
fname = os.path.splitext(file_name)[0]
print(fname + "" + file_name)

if file_name.endswith('.xlsx'):
    read_file = pd.read_excel(file_name)
    df = read_file.to_csv(fname + '.csv', index = None, header=True, encoding = 'ISO-8859-1')
    
elif file_name.endswith('.txt'):
    read_file = pd.read_csv(file_name, error_bad_lines=False)
    df = read_file.to_csv (fname + '.csv', index=None)
    print("Text file")
    
elif file_name.endswith('.pdf'):
    df = tabula.convert_into(file_name, fname + '.csv', pages = 'all')
    print("PDF file")
    
else:
    print("Enter correct file")

df2 = pd.read_csv(fname + '.csv', encoding='ISO-8859-1',
                        sep=',', error_bad_lines=False, names=['documents'])

df2['index'] = np.arange(1, len(df2)+1)
df2 = df2[['index', 'documents']]

# Converting into lower case
df2['documents'] = df2.documents.map(lambda x: x.lower())


# remove empty strings
df2['new_col'] = df2['documents'].astype(str).str[0]
df2["new_col"] = df2['new_col'].str.replace('[^\w\s]','')

nan_value = float("NaN")
#Convert NaN values to empty string

df2.replace("", nan_value, inplace=True)

df2.dropna(subset = ["new_col"], inplace=True)
df2.drop('new_col', inplace=True, axis=1)
df2.to_excel("error.xlsx")


print(df2.head())


