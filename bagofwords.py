from sklearn.feature_extraction.text import CountVectorizer

# veri seti
doc = ["kedi bhaçede","kedi evde"]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(doc)
X.toarray()
vectorizer.get_feature_names_out()