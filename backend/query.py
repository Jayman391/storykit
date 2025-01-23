import babycenterdb
import babycenterdb.filter
from babycenterdb.query import Query
from datetime import datetime
import pandas as pd

def build_query(params : dict) -> pd.DataFrame:
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
  if params['time_delta']:
    comment_filters.append(babycenterdb.filter.TimeDeltaFilter(floor=params['time_delta'][0], ceiling=params['time_delta'][1]))
  if params['ngram_keywords']:
    post_filters.append(babycenterdb.filter.TextFilter(value_list=params['ngram_keywords'].split(',')))
    comment_filters.append(babycenterdb.filter.TextFilter(value_list=params['ngram_keywords'].split(',')))
  if params['groups']:
    post_filters.append(babycenterdb.filter.GroupFilter(value_list=params['groups'].split(',')))
    comment_filters.append(babycenterdb.filter.GroupFilter(value_list=params['groups'].split(',')))

  post_filters.append(babycenterdb.filter.CountryFilter(value_list=['USA']))
  comment_filters.append(babycenterdb.filter.CountryFilter(value_list=['USA']))

  posts_query = Query('posts', post_filters, output_format="df").execute()
  comments_query = Query('comments', comment_filters, output_format="df").execute()

  posts_df = posts_query.documents
  comments_df = comments_query.documents
  # Concatenate DataFrames
  combined_df = pd.concat([posts_df, comments_df], ignore_index=True)
  
  # Convert ObjectId to String
  if '_id' in combined_df.columns:
      combined_df['_id'] = combined_df['_id'].astype(str)
  
  return combined_df  

    