import nltk

nltk.download('punkt_tab')

text = "Natural Language Processing is a branch artificial intelligence."

#Tokenization
from nltk.tokenize import word_tokenize

token = word_tokenize(text)
#print(token)

#Stop-Word Remowal - is, the, on, at, in...

from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
filtered_tokes = [word for word in token if word not in stop_words]

#print(filtered_tokes)

#Lemmatization -> Kelimeyi kök haline getirme
#running -> run

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
# v -> verb -> fiil
# n -> noun -> isim
# a -> adjective -> sıfat
# r -> adverb -> zarf

#print(lemmatizer.lemmatize('running', pos='v'))


# pos tagging 

nltk.download('averaged_perceptron_tagger_eng')
from nltk import pos_tag

pos_tags = pos_tag(filtered_tokes)
#print(pos_tags)

#NER -> Named Entity Recognition

nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk import ne_chunk

tree = ne_chunk(pos_tags)
#print(tree)

# Metin temizleme ve ön işleme
# Lowercasing

text = "Natural Language Processing is a branch artificial intelligence."

text = text.lower()
#print(text)


import re 

text = re.sub(r'[^\w\s]', '', text) #Regex
#print(text)

# Vectorize -> metinleri sayısal ifadelere dönüştürmek
# Bag of words
corpus = [
    "Natural Language Processing is a branch artificial intelligence.",
    "I love studying NLP",
    "Language is a tool for communication.",
    "Language models can understand texts."
]

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

# print(vectorizer.get_feature_names_out())
# print(X.toarray())

# Tf-idf -> metin içindeki önemli kelimeleri ağırlıklandırır.
# tf -> kelime kaç kere geçiyor, idf -> kelimelerin metin içindeki nadirliği

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer2 = TfidfVectorizer()
X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names_out())
print(X2.toarray())

# pipeline =>
# 1- Tokenization - Lowercasing
# 2- Stopwords temizliği
# 3- Lemmatization
# 4- Tf-idf vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır