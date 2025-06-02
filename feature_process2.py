import pandas as pd
import os
def to_df(file_path):
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
            if i%1000000==0:
                print(i)
        df = pd.DataFrame.from_dict(df, orient='index')
        return df
input_root = './data'
review1_name = 'reviews_Books_5.json'
meta1_name = 'meta_Books.json'
review2_name = 'reviews_Toys_and_Games_5.json'
meta2_name = 'meta_Toys_and_Games.json'
review1_path = os.path.join(input_root, review1_name)
meta1_path = os.path.join(input_root, meta1_name)
review2_path = os.path.join(input_root, review2_name)
meta2_path = os.path.join(input_root, meta2_name)
###read data
review1_df = to_df(review1_path)
review2_df = to_df(review2_path)
###choose interested column
review1_df = review1_df[['reviewerID', 'asin','overall','unixReviewTime']]
review2_df = review2_df[['reviewerID', 'asin','overall','unixReviewTime']]
###find common users
c_review1_df = review1_df[ review1_df['reviewerID'].isin(review2_df['reviewerID'].unique())]
c_review2_df = review2_df[ review2_df['reviewerID'].isin(c_review1_df['reviewerID'].unique())]
c_review1_df = c_review1_df.reset_index(drop=True)
c_review2_df = c_review2_df.reset_index(drop=True)
###sort by reviewer and time
c_review1_df = c_review1_df.sort_values(['reviewerID', 'unixReviewTime'])
c_review1_df = c_review1_df.reset_index(drop=True)
c_review2_df = c_review2_df.sort_values(['reviewerID', 'unixReviewTime'])
c_review2_df = c_review2_df.reset_index(drop=True)

import json
######create mapping
common_users = c_review1_df['reviewerID'].unique().tolist()
book_items = c_review1_df['asin'].unique().tolist()
movie_items = c_review2_df['asin'].unique().tolist()
def str2id(inputlist):
    str_id = {}
    m = 1
    for item in inputlist:
        str_id[item] = m
        m = m+1
    return str_id
user2id = str2id(common_users)
book2id = str2id(book_items)
movie2id = str2id(movie_items)
###source:book,1; target:movie,2
source_dict = {}
source_all_behavior = {}
counter = 0
for reviewer_ID, sour_hist in c_review1_df.groupby('reviewerID'):
    source_dict[reviewer_ID] = sour_hist
for reviewerID, tar_hist in c_review2_df.groupby('reviewerID'):
    userid = user2id[reviewerID]
    tar_behavior = tar_hist['asin'].tolist()
    tar_rate = tar_hist['overall'].tolist()
    tar_time = tar_hist['unixReviewTime'].tolist()
    
    sour_behavior = source_dict[reviewerID]['asin'].tolist()
    sour_rate = source_dict[reviewerID]['overall'].tolist()
    sour_time = source_dict[reviewerID]['unixReviewTime'].tolist()
    source_all_behavior[user2id[reviewerID]] = []
    
    training_time = tar_time[-2]
    for m in range(len(sour_behavior)):
        if sour_time[m] < training_time:
            source_all_behavior[ user2id[reviewerID] ].append([ sour_rate[m], book2id[sour_behavior[m]] ])
            counter += 1
print("total source training samples:", counter)
out_root = "./preprocess_data/"
with open( out_root + 'source_pos_neg.json','w' ) as f:
    json.dump(source_all_behavior,f)