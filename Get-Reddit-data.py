import praw
import csv

# Reddit API credentials
reddit = praw.Reddit(client_id='C9m2y3NRDX0fjzW9vGnLKw', 
                     client_secret='TZcYQwlZ1NngXLB_NmSeHNJWyV1zkw', 
                     user_agent='uni_project by /u/mars_million')

# Specify the post ID
post_id = '140xj5s'

# Get the post by its ID
post = reddit.submission(id=post_id)

# Flatten all the comments and replies under the post
post.comments.replace_more(limit=0)
comments_list = post.comments.list()

# Save the comments in a CSV file
with open(f'comments_{post.name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['comment', 'upvotes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for comment in comments_list:
        writer.writerow({'comment': comment.body, 'upvotes': comment.score})
