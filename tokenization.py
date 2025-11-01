import nltk
nltk.download("punkt")

text = "hello world! How are you , hi ..."

text_token = nltk.word_tokenize(text)
sentence_token = nltk.sent_tokenize(text)