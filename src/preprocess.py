import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)