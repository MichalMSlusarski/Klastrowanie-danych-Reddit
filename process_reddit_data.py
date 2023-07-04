import pandas as pd
import numpy as np
import re
import spacy
import string
import textstat
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

with open('stopwords_en.txt', 'r', encoding='utf-8') as f:
    unique_stopwords = ['life', 'people', 'age', 'young', 'old']
    STOP_WORDS = f.read().splitlines()
    STOP_WORDS.extend(unique_stopwords)

negative_words = []
with open('negative.txt', 'r', encoding='utf-8') as g:
    negative_words = g.read().splitlines()

positive_words = []
with open('positive.txt', 'r', encoding='utf-8') as h:
    positive_words = h.read().splitlines()

verbs = []
with open('verbs.txt', 'r', encoding='utf-8') as v:
    verbs = v.read().splitlines()

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('comments_t3_14oszge.csv', sep=',')

def clean_text(text: str) -> str:
    
    text_cleaned = ''

    text = re.sub(r'[^a-zA-Z\s]', '', text)

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
    return df

def df_to_list(df=df, content_col='comment', min_length=10):
    comments = []
    ids = []

    for index, row in df.iterrows():
        comment = row[content_col]
        if len(comment) >= min_length:
            comments.append(comment)
            ids.append(index)
    
    return [comments, ids]

def vectorize_comments(document_list):
    vectorizer = TfidfVectorizer()
    vectorized_docs = vectorizer.fit_transform(document_list)

    pca = PCA(n_components=2)
    reduced_docs = pca.fit_transform(vectorized_docs.toarray())

    return reduced_docs

def DBSCAN_clustering(reduced_docs, epsilon, min):

    dbscan = DBSCAN(eps=epsilon, min_samples=min)
    cluster_labels = dbscan.fit_predict(reduced_docs)

    cluster_counts = {}
    for label in set(cluster_labels):
        cluster_counts[label] = sum(cluster_labels == label)

    for label, count in cluster_counts.items():
        print(f"Cluster {label}: {count} documents")

    return [reduced_docs, cluster_labels]

def kMeans_clustering(reduced_docs, n_clusters):

    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(reduced_docs)

    cluster_counts = {}
    for label in set(cluster_labels):
        cluster_counts[label] = sum(cluster_labels == label)

    for label, count in cluster_counts.items():
        print(f"Cluster {label}: {count} documents")

    return [reduced_docs, cluster_labels]

def draw_viz(reduced_docs, cluster_labels, title):

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(reduced_docs[:, 0], reduced_docs[:, 1], c=cluster_labels, cmap='Set1')

    plt.title(title, fontsize = 20)
    plt.xlabel('')
    plt.ylabel('')
    plt.colorbar(scatter)

    plt.show()

def draw_viz_raw(reduced_docs, title='Rozmieszczenie komentarzy na płaszczyźnie'):

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(reduced_docs[:, 0], reduced_docs[:, 1])
    
    plt.title(title, fontsize = 20)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()

#Top keywords for each cluster
import itertools
from collections import Counter

def aggregate_top_keywords(keywords, cluster_labels, top_n=15):
    cluster_keywords = {}

    for cluster_label in set(cluster_labels):
        cluster_keywords_list = [kw for kw, lbl in zip(keywords, cluster_labels) if lbl == cluster_label]

        flattened_keywords = list(itertools.chain.from_iterable(cluster_keywords_list))
        keyword_counts = Counter(flattened_keywords)
        top_keywords = keyword_counts.most_common(top_n)
        cluster_keywords[cluster_label] = top_keywords

    return cluster_keywords

# import itertools
# from collections import Counter

def print_top_keywords_for_each_cluster(keywords, cluster_labels):
    keyword_data = []
    top_keywords_per_cluster = aggregate_top_keywords(keywords, cluster_labels)

    for cluster_label, top_keywords in top_keywords_per_cluster.items():
        print(f"Cluster {cluster_label}:")
        keyword_data.append([f'CLUSTER {cluster_label}', 0])
        for keyword, count in top_keywords:
            print(f"- {keyword}") #({count} occurrences)")
            keyword_data.append([keyword, count])
        print()
    df = pd.DataFrame(keyword_data, columns=['keyword', 'count'])
    df.to_csv(f'keywords_from_4_clusters.csv', index=False)
    keyword_data.clear()

def remove_words(bag_of_words, words_to_exclude):
    modified_bag_of_words = []

    for keywords in bag_of_words:
        modified_keywords = [word for word in keywords if word not in words_to_exclude]
        modified_bag_of_words.append(modified_keywords)

    return modified_bag_of_words


common_words = ["mistake", "make", "think", "good", "well", "regret", "move", "time", "like", "learn", "let", "bad", "look", "know", "way", "try", "love", "lesson", "forgive", "part", "change", "live", "less", "grow"]
clean_df = clean_all()
doc_list = df_to_list(clean_df, 'comment')
vectors = vectorize_comments(doc_list[0])
n = 4
epsilon = 0.04
min = 6
#draw_viz_raw(vectors)

output = kMeans_clustering(vectors, n)
#output = DBSCAN_clustering(vectors, epsilon, min)
reduced_docs, cluster_labels = output[0], output[1]

comments = doc_list[0]
bag_of_words = [comment.split() for comment in comments]
bag_of_words = remove_words(bag_of_words, common_words)
print_top_keywords_for_each_cluster(bag_of_words, cluster_labels)
#draw_viz(reduced_docs, cluster_labels, f'Grupowanie metodą k-średnich dla {n} grup')
#draw_viz(reduced_docs, cluster_labels, f'Grupowanie DBSCAN dla eps={epsilon} i min={min}')


