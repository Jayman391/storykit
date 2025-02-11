import babycenterdb
import babycenterdb.filter
from babycenterdb.query import Query
from datetime import datetime
import pandas as pd

def build_query(params : dict) -> pd.DataFrame:
  print(params['post_or_comment'])
  post_filters = []
  comment_filters = []
  if params['start_date'] and params['end_date']:
    # check that the string is yyyy-mm-dd, if not then adjust the format
    if len(params['start_date']) != 10:
      year, month, day = params['start_date'].split('-')
      params['start_date'] = f"{year[-4:]}-{month[-2:]}-{day[-2:]}"
      # keep last 4 digits for year, month, day = params['start_date'].split('-')
    if len(params['end_date']) != 10:
      year, month, day = params['end_date'].split('-')
      params['end_date'] = f"{year[-4:]}-{month[-2:]}-{day[-2:]}"

    post_filters.append(babycenterdb.filter.DateFilter(floor=datetime.strptime(params['start_date'], '%Y-%m-%d'), ceiling=datetime.strptime(params['end_date'],'%Y-%m-%d')))
    comment_filters.append(babycenterdb.filter.DateFilter(floor=datetime.strptime(params['start_date'], '%Y-%m-%d'), ceiling=datetime.strptime(params['end_date'], '%Y-%m-%d')))
  if params['comments_range']:
    post_filters.append(babycenterdb.filter.NumCommentsFilter(floor=params['comments_range'][0], ceiling=params['comments_range'][1]))
  if params['ngram_keywords']:
    post_filters.append(babycenterdb.filter.TextFilter(value_list=params['ngram_keywords'].split(',')))
    comment_filters.append(babycenterdb.filter.TextFilter(value_list=params['ngram_keywords'].split(',')))
  if params['groups']:
    post_filters.append(babycenterdb.filter.GroupFilter(value_list=params['groups'].split(',')))
    comment_filters.append(babycenterdb.filter.GroupFilter(value_list=params['groups'].split(',')))

  post_filters.append(babycenterdb.filter.CountryFilter(value_list=['USA']))
  comment_filters.append(babycenterdb.filter.CountryFilter(value_list=['USA']))

  if len(params['post_or_comment']) == 2:
    posts_query = Query('posts', post_filters, output_format="df", limit=params['num_documents']).execute()
    comments_query = Query('comments', comment_filters, output_format="df", limit=params['num_documents']).execute()

    posts_df = posts_query.documents
    posts_df['type'] = 'post'
    comments_df = comments_query.documents
    comments_df['type'] = 'comment'

    # Concatenate DataFrames
    combined_df = pd.concat([posts_df, comments_df], ignore_index=True)
    
    # Convert ObjectId to String
    if '_id' in combined_df.columns:
        combined_df['_id'] = combined_df['_id'].astype(str)
    
    return combined_df  
  elif params['post_or_comment'][0] == 'post':
    query = Query(params['post_or_comment'][0], post_filters, output_format="df", limit=params['num_documents']).execute()
    df = query.documents
    df['type'] = 'post'
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str) 
    return df
  elif params['post_or_comment'][0] == 'comment':
    query = Query(params['post_or_comment'][0], comment_filters, output_format="df", limit=params['num_documents']).execute()
    df = query.documents
    df['type'] = 'comment'
    if '_id' in df.columns:
        df['_id'] = df['_id'].astype(str)
    return df
    