import praw
import csv

# Dostęp do API (ukryty)
reddit = praw.Reddit(client_id='C9m2y3NRDX0fjzW9vGnLKw', 
                     client_secret='TZcYQwlZ1NngXLB_NmSeHNJWyV1zkw', 
                     user_agent='uni_project by /u/mars_million')

# Numer identyfikacyjny posta
post_id = '14oszge'

# Pobranie posta
post = reddit.submission(id=post_id)

# Wypłaszczenie, czyli zrównanie wszystkich komentarzy i odpowiedzi (dehierarchizacja)
post.comments.replace_more(limit=0)
comments_list = post.comments.list()

# Utworzenie CSV i zapisanie danych
with open(f'comments_{post.name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['author', 'comment', 'upvotes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for comment in comments_list:
        writer.writerow({'author': comment.author, 'comment': comment.body, 'upvotes': comment.score})
