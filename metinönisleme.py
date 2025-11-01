# metinlerde bulunan fazla boşlukarı temizleme

text = "Hello,       World!         20235"
text_new = " ".join(text.split())

# küçük har çevirme

text_new = text_new.lower()


# özel karakterleri kaldır
# noktalama işaretleri temizle

import re

# burada ^ ile bunlar haricindekileri temizle demiş olduk
text_new = re.sub(r"[^A-Za-z0-9\s]","",text_new) 
text_new = re.sub("[.,;:?!]","",text_new)


# yazım yanlısları düzlet

from textblob import TextBlob

text = "hello wirld 20235"
text = TextBlob(text).correct()
print(text)

# html ve url etiketlerini kaldır
from bs4 import BeautifulSoup

html_text = "<div>Hello, World 2025</div>"
html_text = BeautifulSoup(html_text,"html.parser").get_text()