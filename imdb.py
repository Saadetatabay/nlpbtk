import pandas as pd
import re

dt = pd.read_csv("IMDB Dataset.csv")
print(dt.columns)

#veri temizleme
dt["review"] = dt["review"].apply(lambda x:x.lower()) #küçük harfe çevrilirdi
dt["review"] = dt["review"].apply(lambda x:re.sub(r"\d+","",x)) #noktalama işarteleri temizlendi
dt["review"] = dt["review"].apply(lambda x:re.sub(r'[.,:?!^#&$-<>]',"",x)) #noktalama işarteleri temizlendi
dt["review"] = dt["review"].apply(lambda x: " ".join([word for word in x.split() if len(word) > 2]))

print(dt.head())
