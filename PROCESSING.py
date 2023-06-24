import pandas as pd
import re
import spacy
import json
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from google.cloud import bigquery
from google.oauth2 import service_account

# ŁADOWANIE WYMAGANYCH PLIKÓW

credentials = service_account.Credentials.from_service_account_file('secret.json')

project_id = 'newonce-178415'
client = bigquery.Client(credentials=credentials, project=project_id)

input_query = ("""
   SELECT DISTINCT *
   FROM content.articles_content
   ORDER BY date""")

df = client.query(input_query).to_dataframe()# Wait for the job to complete.

nlp = spacy.load("pl_core_news_sm")

with open('categories.json', 'r') as file:
    categories = file.read()

CAT_KEYWORDS_DICT = json.loads(categories)

with open('polish.stopwords.txt', 'r', encoding='utf-8') as f:
    STOP_WORDS = f.read().splitlines()

# FUNKCJE

'''Podstawowe czyszczenie tekstu. Usuwane są znaki specjalne, stopwordy, a wielkie litery stają się małe.
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

'''W tej funkcji analizujemy tekst poddany czyszczeniu (text_cleaned) w celu wygenerowania tematów za pomocą modelu LDA (Latent Dirichlet Allocation). 
Jest on trenowany na tym korpusie. Następnie generujemy określoną liczbę tematów (num_topics). 
Dla każdego tematu, pobieramy określoną liczbę słów kluczowych (num_words) i tworzymy listę tematów w postaci krotek (słowo, waga). 
Ostatecznie zwracamy listę wygenerowanych tematów.
'''
def get_topics(text_cleaned: str, num_topics=5, num_words=3) -> list:

    tokens = simple_preprocess(text_cleaned)

    dictionary = corpora.Dictionary([tokens])

    corpus = [dictionary.doc2bow(tokens)]

    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topics = []
    for topic_id in range(num_topics):
        topic_words = lda_model.show_topic(topic_id, topn=num_words)
        topic = [(word, weight) for word, weight in topic_words]
        topics.extend(topic)

    return topics

'''W tej funkcji przerabiamy listę topics na listę keywordów, czyli właściwe to samo tylko
   bez wag przypisanych do każdego słowa.
'''
def get_keywords(topics: list) -> list:

    keywords = []
    keywords = [word for word, _ in topics]

    keywords.sort()
    return keywords

def get_unique_keywords(keywords: list) -> str:

    key_set = set(keywords)
    unique_keywords = ', '.join(key_set)

    return unique_keywords

'''Ta funkcja porównuje słowa kluczowe dla analizowanego tekstu ze słowami uzyskanymi dla klastrów z resarchu na danych 2022.
   Korzystamy z cosine similarity i zwracamy nazwę najbardziej podobnego klastru jako kategorię.
'''
def find_most_similar_category(categories: dict, keywords_list: list) -> str:
    
    documents = list(categories.values())
    documents.append(keywords_list)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(doc) for doc in documents])

    keywords_list_tfidf = tfidf_matrix[-1]  # TF-IDF representation of keywords_list

    similarities = cosine_similarity(keywords_list_tfidf, tfidf_matrix[:-1]).flatten()

    most_similar_category_index = similarities.argmax()
    most_similar_category = list(categories.keys())[most_similar_category_index]
    
    return most_similar_category

'''Ta funkcja analizuje sentyment każdego ze zdań w tekście korzystając z przetrenowanego modelu.
   Zwraca średnią dla całego dokumentu
'''
def get_sentiment(text: str) -> float:

    from sentimentpl.models import SentimentPLModel

    import sacremoses
    model = SentimentPLModel(from_pretrained='latest')

    sentence_sentiments = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        value = model(sentence).item()
        sentence_sentiments.append(value)

    sentiment = sum(sentence_sentiments) / len(sentences)
    round(sentiment, 5)

    return sentiment

'''Rolą tej funkcji jest zwrócenie czytelności tekstu w oparciu o indeks Gunninga.
   Wynikiem jest liczba oznaczająca, w której klasie (USA) musi być przeciętny uczeń aby przeczytać tekst.
'''
def get_readability(text: str) -> float:

    textstat.set_lang('pl')
    readability = textstat.gunning_fog(text)

    return readability

'''Poniższe funkcje przekształcają listy na format czytleny dla BigQuery
'''
def reformat_topics(topics: list) -> str:

    topics_str = ' + '.join(f"(\"{word}\",{value})" for word, value in topics)

    return topics_str

def reformat_keywords(keywords: list) -> str:

    keywords_string = ', '.join(keywords)

    return keywords_string


# MAIN

'''W funkcji main wywołujemy poprzednio zadeklarowane funkcje, przekazując kolejno zwracane outputy do dalszych funkcji.
   Na końcu tworzymy nowy rząd dla tabeli master i nadpisujemy go.
'''   
def main():

    date, author, title, text, link = map(str, df.iloc[0, :5])
    char_count = len(text)
    text_cleaned = clean_text(text)
    topics = get_topics(text_cleaned)
    keywords = get_keywords(topics)
    category = find_most_similar_category(CAT_KEYWORDS_DICT, keywords)
    sentiment = get_sentiment(text)
    readability = get_readability(text)
    unique_keywords = get_unique_keywords(keywords)

    topics = reformat_topics(topics)
    keywords = reformat_keywords(keywords)

    new_row = {
        'date': date,
        'link': link,
        'author': author,
        'title': title,
        'text': text,
        'text_clean': text_cleaned,
        'char_count': char_count,
        'topics': topics,
        'keywords': keywords,
        'unique_keywords': unique_keywords,
        'category': category,
        'sentiment': sentiment,
        'readability': readability
    }

    columns = ', '.join(new_row.keys())
    values = ', '.join([
        str(value) if isinstance(value, (int, float))
        else f"'{value}'"
        for value in new_row.values()
    ])

    out_query = f"""
    INSERT INTO newonce-178415.content.articles_analytics_1 ({columns})
    VALUES ({values})
    """

    q = client.query(out_query)
    print(q.result())


main()