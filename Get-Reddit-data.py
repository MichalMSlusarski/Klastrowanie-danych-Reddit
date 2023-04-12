import praw
import csv

# Reddit API credentials
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID', 
                     client_secret='YOUR_CLIENT_SECRET', 
                     user_agent='YOUR_USER_AGENT')

# Specify the post ID
post_id = 'POST_ID'

# Get the post by its ID
post = reddit.submission(id=post_id)

# Flatten all the comments and replies under the post
post.comments.replace_more(limit=None)
comments_list = post.comments.list()

# Save the comments in a CSV file
with open('comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['comment', 'upvotes']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for comment in comments_list:
        writer.writerow({'comment': comment.body, 'upvotes': comment.score})
