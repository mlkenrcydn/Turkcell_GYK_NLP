# pipeline =>
# 1- Tokenization - Lowercasing
# 2- Stopwords temizliği
# 3- Lemmatization
# 4- Tf-idf vektörleştirme
# 5- Feature isimlerini ve arrayi ekrana yazdır

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import string

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def nlp_piepline(corpus):

    # 1- Tokenization - Lowercasing
    tokenized_corpus = []
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        tokenized_corpus.append(tokens)
    #print(tokenized_corpus)
    
    # 2- Stopwords temizliği
    stop_words = set(stopwords.words('english'))
    cleaned_corpus = []
    for tokens in tokenized_corpus:
        filtered = [word for word in tokens if word not in stop_words]
        cleaned_corpus.append(filtered)
    #print(cleaned_corpus)
    
    # 3- Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = []
    for tokens in cleaned_corpus:
        lemmatized = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
        lemmatized_corpus.append(' '.join(lemmatized))
    print(lemmatized_corpus)

    # 4- Tf-idf vektörleştirme
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lemmatized_corpus)

    # 5. Sonuçları yazdır
    print("Feature İsimleri (TF-IDF Sütun Etiketleri):\n", vectorizer.get_feature_names_out())
    print("\nTF-IDF Array:\n", tfidf_matrix.toarray())

corpus = [
        "Natural Language Processing is a branch of artificial intelligence.",
        "I love studying NLP.",
        "Language is a tool for communication.",
        "Language models can understand texts.",
        "Text classification is a common NLP task.",
        "Tokenization splits text into words or subwords.",
        "Stemming and lemmatization help normalize words.",
        "Named Entity Recognition extracts important entities.",
        "Sentiment analysis can detect emotions in text.",
        "Chatbots use NLP to interact with users.",
        "Word embeddings map words into vector space.",
        "Transformers have revolutionized natural language processing.",
        "BERT and GPT are powerful language models.",
        "Preprocessing is essential before vectorization.",
        "NLP techniques are widely used in social media analysis.",
        "POS tagging identifies parts of speech in a sentence.",
        "Stopwords are often removed during preprocessing.",
        "TF-IDF highlights important words in a document.",
        "Language generation is a challenging NLP problem.",
        "Deep learning has improved the performance of NLP models."
    ]

nlp_piepline(corpus)