import pandas as pd
import matplotlib.pyplot as plt

#principle exponent analysis : dimension reduction
from sklearn.decomposition import PCA
from gensim.models import Word2Vec,FastText
from gensim.utils import simple_preprocess

doc = ["Köpek çok tatlı bir hayvandır.",
       "Köpekler evcil hayvanlardır."
       "Kediler genellikle bağımsız hareket etmeyi severler.",
       "Köpekler sadık ve insan canlısı hayvanlardır."
       "Hayvanlar insanşar için iyi arkadaşlardır."]

#kelimeleri küçük harfe çevirip tokenlara ayırdı
tokenized_sentence = [simple_preprocess(sentence) for sentence in doc]

#vector_size = 50 her kelime için üreteceği vektörün gömülme boyutu. yani her kelime 50 sayıdan oluşan bir liste ile temsil edilecek
#window bir keimenin bağlamını belirlemek için sağında ve solunda beş kelimeye bakacak
#min_count modelin veri setinde sadece bir kere bile geçen kelimeleri dahil eder
#sg = 0 (CBOW) sg = 1 (Skip-gram)
word2vec_model = Word2Vec(sentences=tokenized_sentence,vector_size=50,window=5,min_count=1,sg=1)
fasttext_model = FastText(sentences=tokenized_sentence,vector_size=50,window=5,min_count=1,sg=1)
#word2vec_model.wv['köpek'] 50 lik array döndü

word_wv = word2vec_model.wv

#dict key:kelime value :indeks
word_dict = word2vec_model.wv.key_to_index

#kelimeleri aldık
words = list(word2vec_model.wv.key_to_index)
#her bir kelimenin 50 boyutlu vektörünü numpy arrayini alır
vectors = [word_wv[word] for word in words]


#50 boyutu 3d ye çevirmek için
pca = PCA(n_components=3)
reduced_vektor = pca.fit_transform(vectors)

#görselleştirme

fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111,projection="3d")
ax.scatter(reduced_vektor[:,0],reduced_vektor[:,1],reduced_vektor[:,2])

#kelimeleri etiketleme
for i,word in enumerate(words):
    #xyz kordinatlarını o kelime ile etiketliyoruz
    ax.text(reduced_vektor[i,0],reduced_vektor[i,1],reduced_vektor[i,2],word)

plt.show()
