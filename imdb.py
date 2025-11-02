import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

dt = pd.read_csv("/home/satabay/nlpbtk/nlpbtk/IMDB Dataset.csv")
# print(dt.columns)
# print(dt.head())

#veri temizleme
dt["review"] = dt["review"].apply(lambda x:x.lower()) #küçük harfe çevrilirdi
dt["review"] = dt["review"].apply(lambda x:re.sub(r"\d+","",x)) #noktalama işarteleri temizlendi
dt["review"] = dt["review"].apply(lambda x:re.sub(r'[.,:?!^#&$-<>+_*()=]',"",x)) #noktalama işarteleri temizlendi
dt["review"] = dt["review"].apply(lambda x: " ".join([word for word in x.split() if len(word) > 2]))

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(dt["review"][0:75])
vektor_temsili = x.toarray()
feature_names = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(vektor_temsili,columns=feature_names)
print(df_bow.head(15))
word=x.sum(axis=0).A1 #A1 ile 2D matrisi 1D hale getirdik
word_freq = dict(zip(feature_names, word))
print(Counter(word_freq).most_common(5))