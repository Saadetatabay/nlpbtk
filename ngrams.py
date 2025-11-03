from sklearn.feature_extraction.text import CountVectorizer

doc = ["Bu calisma ngram calismasıdır",
       "bu calisma dogal dil isleme calismasıdır"]

vectorizer_unigram = CountVectorizer(ngram_range=(1,1))
vectorizer_bigram = CountVectorizer(ngram_range=(2,2))
vectorizer_trigram = CountVectorizer(ngram_range=(3,3))

x_unigram = vectorizer_unigram.fit_transform(doc)
x_unigram = x_unigram.toarray()
vectorizer_unigram.get_feature_names_out()

x_bigram = vectorizer_bigram.fit_transform(doc)
x_bigram = x_bigram.toarray()
vectorizer_bigram.get_feature_names_out()