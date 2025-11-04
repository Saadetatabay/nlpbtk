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
# dt["review"] = dt["review"].apply(lambda x: re.sub(r":;.,-_=[]()!'^#½%&","",x))
#\w word character \s space tab falan başındaki r ile bunlar haricindekiler "" ile değiştir diyoruz diliyoruz yani
dt["review"] = dt["review"].apply(lambda x: re.sub(r"[^\w\s]","",x))
dt["review"] = dt["review"].apply(lambda x:" ".join([w for w in x.split() if len(w) > 2]))

#cümleleri tokenlara ayırır 
tokenized_doc = [simple_preprocess(x) for x in dt["review"]]

#her kelime 50 boyutlu bir vektörle temsil edilecek wv ile word vektörü çekiyor sadece
word_vec = Word2Vec(sentences=tokenized_doc,
                    vector_size=50,
                    window=5,
                    min_count=1,
                    sg=0).wv

#indeks to key her bir kelimeyi alır  listeye koay indeks ile o keliemye erişriz

words = word_vec.index_to_key[:500]
vektors = [word_vec[word] for word in words]

print(word_vec.index_to_key[0])

print(word_vec["her"].size)

kmeans = KMeans(n_clusters=2)
kmeans.fit(vektors) #keliemekrin 50lik vektörlerini gönderdik
kume_etiketleri = kmeans.labels_

#PCA ile 50->2
pca = PCA(n_components=2)
reduced_vec = pca.fit_transform(vektors)

#veri görselleştirme

plt.figure()
#c = kumeetiketleri ile her noktanuın rengini 0 1 olmasına göre renklendirir 
plt.scatter(reduced_vec[:,0],reduced_vec[:,1],c=kume_etiketleri,cmap="viridis")

#kelimeleri labellama
for i,word in enumerate(words):
    plt.text(reduced_vec[i,0],reduced_vec[i,1],word)



plt.show()