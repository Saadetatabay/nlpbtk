import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = stopwords.words("english")
text = "there are some examples of handling stop words from some texts."
[x for x  in text.split() if x not in stop_words]

