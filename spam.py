import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from collections import Counter

df = pd.read_csv("/home/satabay/nlpbtk/datasets/spam.csv",encoding='latin-1')

df["v2"] = df["v2"].apply(lambda x:x.lower())
df["v2"] = df["v2"].apply(lambda x: re.sub(r"\d+","",x))
df["v2"] = df["v2"].apply(lambda x: re.sub(r"[:.;,?!<-_>^+$]","",x))

stopword = stopwords.words("english")
df["v2"] = df["v2"].apply(lambda x:" ".join([w for w in x.split() if w not in stopword]))

tf_idf_vectorizer = TfidfVectorizer()
x=tf_idf_vectorizer.fit_transform(df["v2"])
feature_names = tf_idf_vectorizer.get_feature_names_out()
vektor_temsili = x.toarray() #bu her satırda her kelimenin tf_idf skoru 2D

df_bow = pd.DataFrame(vektor_temsili,columns=feature_names)
#print(df_bow.head())

# tüm mesajlardaki her bir kelimenin toplam TF_idf skorunu veir
tf_idf_deger = x.sum(axis=0).A1 
dict_word = dict(zip(feature_names,tf_idf_deger))
#print(dict_word.keys())
print(Counter(dict_word).most_common(5))
