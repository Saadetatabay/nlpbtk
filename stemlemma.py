import nltk
from nltk.stem import PorterStemmer

nltk.download("wordnet")

# porter stemmer nesnesi oluşturduk
stemmer = PorterStemmer()

words = ["running","runner","ran","runs","better","go","went"]

#kelimelerin stemlerini buluyoruz
[stemmer.stem(w) for w in words]

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

[lemmatizer.lemmatize(w,pos="v") for w in words]