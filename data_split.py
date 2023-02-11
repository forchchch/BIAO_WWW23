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
# meta1_name = 'meta_Electronics.json'
review2_name = 'reviews_Toys_and_Games_5.json'
# meta2_name = 'meta_Clothing_Shoes_and_Jewelry.json'
review1_path = os.path.join(input_root, review1_name)
#meta1_path = os.path.join(input_root, meta1_name)
review2_path = os.path.join(input_root, review2_name)
#meta2_path = os.path.join(input_root, meta2_name)
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
train_set = []
valid_set = []
test_set = []
meta_indices = []
meta_train_indices = []
meta_num = 1000
source_dict = {}
for reviewer_ID, sour_hist in c_review1_df.groupby('reviewerID'):
    source_dict[reviewer_ID] = sour_hist
training_num = 0
meta_recorder = 0
for reviewerID, tar_hist in c_review2_df.groupby('reviewerID'):
    userid = user2id[reviewerID]
    tar_behavior = tar_hist['asin'].tolist()
    tar_rate = tar_hist['overall'].tolist()
    tar_time = tar_hist['unixReviewTime'].tolist()
    sour_behavior = source_dict[reviewerID]['asin'].tolist()
    #sour_behavior = group_review_source[reviewerID]['asin'].tolist()
    sour_rate = source_dict[reviewerID]['overall'].tolist()
    sour_time = source_dict[reviewerID]['unixReviewTime'].tolist()
    target_behavior_num = len(tar_behavior)
    for m in range(len(tar_behavior)):
        candidate = movie2id[tar_behavior[m]]
        candidate_time = tar_time[m]
        rate = tar_rate[m]
        tar_his = []
        sour_his = []
        for j in range(m):
            if tar_rate[j] > 3:
                tar_his.append(movie2id[tar_behavior[j]])
        for j in range(len(sour_behavior)):
            if sour_time[j]<=candidate_time and sour_rate[j]>3:
                sour_his.append(book2id[sour_behavior[j]])
        piece = [userid, candidate, rate, tar_his, sour_his]
        if m<target_behavior_num-2:
            train_set.append(piece)
            training_num += 1
            if (m == target_behavior_num-3) and meta_recorder<meta_num:
                meta_indices.append(training_num-1)
                meta_recorder += 1
            else:
                meta_train_indices.append(training_num-1)
        elif m==target_behavior_num-2:
            valid_set.append(piece)
        else:
            test_set.append(piece)
print(len(train_set))
print(len(valid_set))
print(len(test_set))
print("meta dataset number:",len(meta_indices))
print("meta train dataset number:",len(meta_train_indices))
# train_set[30000]

target_dict = {}
for reviewer_ID, tar_hist in c_review2_df.groupby('reviewerID'):
    target_dict[reviewer_ID] = tar_hist
source_trainset = []
for reviewer_ID, sour_hist in c_review1_df.groupby('reviewerID'):
    sour_behavior = sour_hist['asin'].tolist()
    sour_rate = sour_hist['overall'].tolist()
    sour_time = sour_hist['unixReviewTime'].tolist()
    tar_behavior = target_dict[reviewer_ID]['asin'].tolist()
    tar_time_list = target_dict[reviewer_ID]['unixReviewTime'].tolist()
    tar_rate = target_dict[reviewer_ID]['overall'].tolist()
    user_train_time = tar_time_list[-2]
    #print("time:",user_train_time)
    for m in range(len(sour_behavior)):
        if sour_time[m]<user_train_time:
            tar_his = []
            sour_his = []
            sc_time = sour_time[m]
            for j in range(len(tar_behavior)):
                if tar_time_list[j]<=sc_time and tar_rate[j]>3:
                    tar_his.append(movie2id[tar_behavior[j]])
            for j in range(m):
                if sour_rate[j] > 3:
                    sour_his.append(book2id[sour_behavior[j]])
            piece = [user2id[reviewer_ID], book2id[sour_behavior[m]], sour_rate[m],tar_his,sour_his]
            source_trainset.append(piece)
print(len(source_trainset))

target_behavior_dict = {}
source_behavior_dict = {}
interaction1 = 0
interaction2 = 0
for reviewer_ID, sour_hist in c_review1_df.groupby('reviewerID'):
    sour_behavior = sour_hist['asin'].tolist()
    interaction1 = max(interaction1, len(sour_behavior))
    sour_rate = sour_hist['overall'].tolist()
    sour_time = sour_hist['unixReviewTime'].tolist()
    source_behavior_dict[user2id[reviewer_ID]] = []
    tar_time_list = target_dict[reviewer_ID]['unixReviewTime'].tolist()
    user_train_time = tar_time_list[-2]
    for m in range(len(sour_behavior)):
        if sour_time[m]< user_train_time and sour_rate[m]>3:
            source_behavior_dict[user2id[reviewer_ID]].append( book2id[sour_behavior[m]] )

for reviewer_ID, tar_hist in c_review2_df.groupby('reviewerID'):
    tar_behavior = tar_hist['asin'].tolist()
    interaction2 = max(interaction2, len(tar_behavior))
    tar_rate = tar_hist['overall'].tolist()
    tar_time = tar_hist['unixReviewTime'].tolist()
    target_behavior_dict[user2id[reviewer_ID]] = []
    for m in range(len(tar_behavior)):
        if m<len(tar_behavior)-2 and tar_rate[m]>3:
            target_behavior_dict[user2id[reviewer_ID]].append( movie2id[tar_behavior[m]] )

print(interaction1,interaction2)
source_maxlen = 0
target_maxlen = 0
for user, his in source_behavior_dict.items():
    source_maxlen = max(source_maxlen, len(his))
for user, his in target_behavior_dict.items():
    target_maxlen = max(target_maxlen, len(his))
print(source_maxlen)
print(target_maxlen)

id2user = {key:value for value,key in user2id.items()}

import json
out_root = "./preprocess_data/"
with open( out_root + 'train_set.json','w' ) as f:
    json.dump(train_set,f)
with open( out_root + 'valid_set.json','w' ) as f:
    json.dump(valid_set,f)
with open( out_root+'test_set.json','w' ) as f:
    json.dump(test_set,f)
with open( out_root+'source_training.json','w' ) as f:
    json.dump(source_trainset,f)
with open( out_root+'source_user_behavior.json','w' ) as f:
    json.dump(source_behavior_dict,f)
with open( out_root+'target_user_behavior.json','w' ) as f:
    json.dump(target_behavior_dict,f)
    
with open( out_root+'meta_indices.json','w' ) as f:
    json.dump(meta_indices,f)
with open( out_root+'meta_train_indices.json','w' ) as f:
    json.dump(meta_train_indices,f)

user_set = c_review1_df['reviewerID'].unique().tolist()
source_item_set = c_review1_df['asin'].unique().tolist()
target_item_set = c_review2_df['asin'].unique().tolist()

with open( out_root+'source_item.json','w' ) as f:
    json.dump(book2id,f)
with open( out_root+'target_item.json','w' ) as f:
    json.dump(movie2id,f)
with open( out_root+'users.json','w' ) as f:
    json.dump(user2id,f)
print("user num:",len(user2id))
print("source item num:", len(book2id))
print("target item num:", len(movie2id))

