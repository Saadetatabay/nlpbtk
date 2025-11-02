import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

doc = ["Köpek çok tatlı hayvandır.",
       "Köpek ve kuşlar çok tatlı hayvanlardır.",
       "Inekler süt üretirler."]

tfidf_vecortizer = TfidfVectorizer()
x = tfidf_vecortizer.fit_transform(doc)
feature_names = tfidf_vecortizer.get_feature_names_out()
vector_temsili = x.toarray()
df_tfidf = pd.DataFrame(vector_temsili,columns=feature_names)