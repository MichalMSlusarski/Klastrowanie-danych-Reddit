import pandas as pd
import numpy as np
import re
import spacy
import string
import textstat
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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

def clean_all():
    df['comment'] = df['comment'].apply(clean_text)
    print(df.head)
    return df

def df_to_list(df=df, content_col='comment'):
    comments = []
    ids = []

    for index, row in df.iterrows():
        comment = row[content_col]
        comments.append(comment)
        ids.append(index)

    return [comments, ids]

def generate_similarity_matrix(document_list):
    global vectorizer

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(document_list)
    cosine_similarities = cosine_similarity(tfidf_matrix)

    return cosine_similarities

def perform_document_clustering(df, column_name, num_clusters):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(df[column_name])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(similarity_matrix)
    df['cluster'] = cluster_labels

    return df

def visualize_clusters(df):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df.index, df.index, c=df['cluster'], cmap='viridis')
    plt.title('Document Clustering')
    plt.xlabel('Document Index')
    plt.ylabel('Document Index')
    plt.legend(handles=scatter.legend_elements()[0], labels=range(3), title='Cluster')
    plt.show()

#Top keywords for each cluster
import itertools
from collections import Counter

def aggregate_top_keywords(keywords, cluster_labels, top_n=30):
    cluster_keywords = {}

    for cluster_label in set(cluster_labels):
        # Get the keywords associated with the current cluster label
        cluster_keywords_list = [kw for kw, lbl in zip(keywords, cluster_labels) if lbl == cluster_label]

        # Flatten the list of keywords
        flattened_keywords = list(itertools.chain.from_iterable(cluster_keywords_list))
        keyword_counts = Counter(flattened_keywords)
        top_keywords = keyword_counts.most_common(top_n)
        cluster_keywords[cluster_label] = top_keywords

    return cluster_keywords

def print_top_keywords_for_each_cluster(keywords, cluster_labels):
    keyword_data = []
    top_keywords_per_cluster = aggregate_top_keywords(keywords, cluster_labels)

    # Print the top keywords for each cluster
    for cluster_label, top_keywords in top_keywords_per_cluster.items():
        print(f"Cluster {cluster_label}:")
        keyword_data.append([f'CLUSTER {cluster_label}', 0, 0])
        for keyword, count in top_keywords:
            print(f"- {keyword} ({count} occurrences)")
            of_all = count / 22962 #hardcoded number of all keywords in all clusters
            keyword_data.append([keyword, count, of_all])
        print()
    df = pd.DataFrame(keyword_data, columns=['keyword', 'count', 'of_all'])
    df.to_csv(f'keywords_from_6_clusters.csv', index=False)
    #keyword_data.clear()


clean_df = clean_all()
x = perform_document_clustering(clean_df, 'comment', 3)
visualize_clusters(x)

