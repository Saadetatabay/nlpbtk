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

