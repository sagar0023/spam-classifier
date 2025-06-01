import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Filter alphanumeric, remove stopwords and punctuation, and stem
    filtered = [
        ps.stem(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(filtered)