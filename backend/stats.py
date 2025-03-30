import pandas as pd

def compute_statistics(data):
    posts = data[data['type'] == 'post']
    comments = data[data['type'] == 'comment']

    avg_post_length = posts['text'].str.split().apply(len).mean()
    num_unique_posters = posts['author'].nunique()
    avg_num_comments = posts['num_comments'].mean()

    avg_comment_length = comments['text'].str.split().apply(len).mean()
    num_unique_commenters = comments['author'].nunique()

    return {
        'avg_post_length': avg_post_length,
        'num_unique_posters': num_unique_posters,
        'avg_num_comments': avg_num_comments,
        'avg_comment_length': avg_comment_length,
        'num_unique_commenters': num_unique_commenters
    }