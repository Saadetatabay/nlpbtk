import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import re
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

dt = pd.read_csv("/home/satabay/nlpbtk/datasets/IMDB Dataset.csv")

dt["review"] = dt["review"].apply(lambda x: x.lower())
dt["review"] = dt["review"].apply(lambda x: re.sub(r"[\d+]","",x))
dt["review"] = dt["review"].apply(lambda x: re.sub(r":;.,-_=[]()!'^#½%&","",x))
#\w word character \s space tab falan başındaki r ile bunlar haricindekiler "" ile değiştir diyoruz diliyoruz yani
dt["review"] = dt["review"].apply(lambda x: re.sub(r"[^\w\s]","",x))
dt["review"] = dt["review"].apply(lambda x:" ".join([w for w in x.split() if len(w) > 2]))

tokenized-doc = [simple_preprocess(x) for x in dt["review"]]
