

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:56:51 2022

@author: 33652
"""

#Dans sterminal: streamlit run api_streamlit.py

from pathlib import Path
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.write('''
# Application web pour prédire la catégorie d'un article publié
Cette application prédit si un article sera peu partagé, ou très partagé
''')

st.sidebar.header("Paramètres d'entrée")

def user_input():
    average_token_lengths=st.sidebar.slider('Length of article',0.0,8.0,4.5) #min,max,default_value
    self_references_avg_shares=st.sidebar.slider('Average share of referenced articles',0.0,843300.0,6401.7)
    LDA_02=st.sidebar.slider('Closeness to LDA topic 2',0.0,0.9,0.2)
    kw_avg_avg=st.sidebar.slider('Average keywords of average shares',0.0,43567.6,3135.8)
    kw_max_avg=st.sidebar.slider('Average keywords (max shares)',0.0,298400.0,5657.2)
    data={'average_token_lengths': average_token_lengths,
          'self_references_avg_shares': self_references_avg_shares,
          'LDA_02': LDA_02,
          'kw_avg_avg': kw_avg_avg,
          'kw_max_avg': kw_max_avg
          }   #Dictionnaire contenant les paramètres à envoyer
    parametres=pd.DataFrame(data,index=[0])
    return parametres

df=user_input()

st.subheader('On veut trouver la catégorie de cet article:')
st.write(df)
    
#Récupérer dataset

dataset=pd.read_csv('OnlineNewsPopularity.csv')
dfmodel=dataset.copy()

l = list()
for n in dfmodel[' shares']:
    if n >= 3000:
        l.append(1)
    else:
        l.append(0)

dfmodel["popularity"] = l
#st.write(dataset)
dfapi = dfmodel.loc[:,[" average_token_length"," kw_avg_max", " kw_avg_avg"," self_reference_avg_sharess", " LDA_02","popularity"]]
st.dataframe(dfapi)
Xnew = dfapi.drop(["popularity"], axis = 1)
Ynew = dfapi["popularity"]
X=Xnew.astype('int')
y=Ynew.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

#modèle 
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

#prédiction
prediction=rfc.predict(df)
st.subheader("la catégorie de l'article est la suivante:")
st.write(prediction)


    