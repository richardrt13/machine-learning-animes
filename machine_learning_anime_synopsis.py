import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')

dataset = 'https://raw.githubusercontent.com/richardrt13/machine-learning-animes/master/animes.csv'

df = pd.read_csv(dataset)

df = df.drop_duplicates() #limpando duplicatas
df = df.drop(['uid','img_url','link','aired'], axis=1)
df['genre'] = df['genre'].str.replace('[','').str.replace(']','').str.replace("'",'').str.replace('[','').str.replace(',','|').str.replace(' ','')

generos = df.genre.str.get_dummies()

df_animes = pd.concat([df, generos], axis=1)

df_filtrado = df_animes.query("Shounen == 1")

df_filtrado = df_filtrado.reset_index(drop= True)

#Localizando o anime no qual vamos realizar a previsão com base na sinopse

df_filtrado[df_filtrado['title'].astype(str).str.contains('Naruto')]

#Iniciando o processo de Bag of Words limpando a sinopse

sw = stopwords.words('english')

def preprocess_text(text):
  
    tokens = word_tokenize(str(text))
    sw = stopwords.words('english')
    msg_wo_sw = [word for word in tokens if word not in sw]
    return msg_wo_sw

df_filtrado['clean_text'] = df_filtrado['synopsis'].apply(preprocess_text)

all_texts = df_filtrado.iloc[:, -1].apply(lambda x: " ".join(x))

text =  df_filtrado.iloc[43:44, -1].apply(lambda x: " ".join(x))

#Treinando o modelo com as palavras

from sklearn.feature_extraction.text import CountVectorizer
countVectorizer = CountVectorizer()

countVectorizer.fit(text)

cv_result = countVectorizer.transform(text)

countVectorizer_df = pd.DataFrame(cv_result.todense(), columns = countVectorizer.get_feature_names())

all_text_cv_result = countVectorizer.transform(all_texts)

all_text_countVectorizer_df = pd.DataFrame(all_text_cv_result.todense(), columns = countVectorizer.get_feature_names())

df_f = pd.concat([countVectorizer_df, all_text_countVectorizer_df], axis=0)

#Agrupando por similaridade de cosseno

from sklearn.metrics.pairwise import cosine_similarity

r = cosine_similarity(df_f)

df_f['result'] = r[0].T

df_f = df_f.iloc[1:]

df_f = df_f.reset_index(drop= True)
df_f.sort_values(by='result', ascending=False, inplace=True)

df_result = pd.concat([df_filtrado, df_f], axis=1)
df_result = df_result.query("score >= 7")
df_result = df_result.query("members >= 400000")
df_result.sort_values(by='result', ascending=False, inplace=True)
df_result[['title','result','score','members']].head(50)





