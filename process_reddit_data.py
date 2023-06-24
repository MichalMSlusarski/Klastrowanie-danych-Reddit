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

def calculate_similarity_matrix(df=df, column_name='comment'):
    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(df[column_name])

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix

def generate_similarity_rank(cosine_similarities, labels, min_strength):
    data = pd.DataFrame()
    # Print the cosine similarity matrix
    for i in tqdm(range(len(cosine_similarities))):
        for j in range(i + 1, len(cosine_similarities)):
            # print(f"Similarity between list {i} and list {j}: {cosine_similarities[i][j]}")
            tmp = pd.DataFrame()
            if cosine_similarities[i][j] > min_strength and labels[i] != labels[j]:
                tmp['content_1'] = [labels[i]]
                tmp['content_2'] = [labels[j]]
                tmp['similarity'] = [cosine_similarities[i][j]]
                data = data.append(tmp)

    data = data.sort_values(by=['similarity'], ascending = False)

    return data

def cluster_data(document_list, epsilon, min):
    # Fit and transform the vectorizer on the document list
    vectorized_docs = vectorizer.fit_transform(document_list)

    # Apply dimensionality reduction to reduce the vectorized documents to two or three dimensions
    pca = PCA(n_components=2)  # or n_components=3 for 3D visualization
    reduced_docs = pca.fit_transform(vectorized_docs.toarray())

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=min)
    cluster_labels = kmeans.fit_predict(reduced_docs)

    #dbscan = DBSCAN(eps=epsilon, min_samples=min)  # Adjust the values of eps and min_samples as needed
    #cluster_labels = dbscan.fit_predict(reduced_docs)
    #cluster_labels = np.unique(cluster_labels)

    cluster_counts = {}
    for label in set(cluster_labels):
        cluster_counts[label] = sum(cluster_labels == label)

    for label, count in cluster_counts.items():
        print(f"Cluster {label}: {count} documents")

    return [reduced_docs, cluster_labels]

def draw_viz(reduced_docs, cluster_labels, title):
    # Visualize the clustered data
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(12, 12))
    scatter = ax.scatter(reduced_docs[:, 0], reduced_docs[:, 1], c=cluster_labels, cmap='Set1')

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar(scatter)

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


clean_all()

