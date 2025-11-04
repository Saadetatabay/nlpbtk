from transformers import AutoModel , AutoTokenizer

model_name = "bert-base-uncased"
#tokenizasyon(ayırma),sözlük eşleme ve sayısallaştırma
tokenizer = AutoTokenizer().from_pretrained(model_name)

#model transformer katmanlarında işler bağlamsal anlamı temsil eden vektörleri üretir
model = AutoModel.from_pretrained(model_name)

text = "Transformers can be used for natural language processing"

#çıktı pytoarch ile return edilir
inputs = tokenizer(text, return_tensor = "pt")