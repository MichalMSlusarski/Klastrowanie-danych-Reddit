import pandas as pd
import re
import spacy
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.corpus import stopwords

with open('stopwords_en.txt', 'r', encoding='utf-8') as f: #stopwords z nltk, bez koniecznosci pobierania modułu
    STOP_WORDS = f.read().splitlines()

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('comments.csv', sep=',')

# FUNKCJE

'''Podstawowe czyszczenie pojedynczego komentarza. Usuwane są znaki specjalne, stopwordy, a wielkie litery stają się małe.
   Używając modułu SpaCy nlp, dokonujemy lemmatyzacji, czyli sprowadzenia wyrazu do formy podstawowej.
'''
def clean_text(text: str) -> str:
    
    text_cleaned = ''

    text = re.sub(r'[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]', '', text)

    tokens = text.split()
    tokens = " ".join([i for i in text.lower().split()])
    tokens = nlp(tokens)

    clean_tokens = []

    for word in tokens:
        if word.lemma_ not in STOP_WORDS:
            clean_tokens.append(word.lemma_)

    text_cleaned = ' '.join(clean_tokens)
    text_cleaned = str(text_cleaned)

    return text_cleaned

x = clean_text('This is a modified country suv. I like it. Is this a correct statements? No me wonders not.')
print(x)

