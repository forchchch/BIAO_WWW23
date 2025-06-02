import pandas as pd
import os
import json
input_root = './data'
out_root = './preprocess_data/'
item1_path = 'source_item.json'
item2_path = 'target_item.json'
meta1_name = 'meta_Books.json'
meta2_name = 'meta_Toys_and_Games.json'

item1 = json.load( open(out_root + item1_path,'r') )
item2 = json.load( open(out_root + item2_path,'r') )

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
meta1_df = to_df(os.path.join(input_root, meta1_name))
meta2_df = to_df(os.path.join(input_root, meta2_name))
c_meta1_df = meta1_df[ meta1_df['asin'].isin(item1.keys()) ]
c_meta2_df = meta2_df[ meta2_df['asin'].isin(item2.keys()) ]
c_meta1_df = c_meta1_df.reset_index(drop=True)
c_meta2_df = c_meta2_df.reset_index(drop=True)
c_meta1_df.to_csv(out_root + 'source_item.csv')
c_meta2_df.to_csv(out_root + 'target_item.csv')
def obtain_item_cate(df):
    item_cate = {}
    for m in range(len(df)):
        cates = df['categories'][m][0]
        if len(cates) == 1:
            item_cate[df['asin'][m]] = 'others'
        else:
            item_cate[df['asin'][m]] = cates[1]
    return item_cate
s_item_cate = obtain_item_cate(c_meta1_df)       
t_item_cate = obtain_item_cate(c_meta2_df)    
def f_cate2id(item2cate):
    cate2id = {}
    m = 1
    for item, cate in item2cate.items():
        if cate not in cate2id.keys():
            cate2id[cate] = m
            m += 1
    return cate2id
s_cate2id = f_cate2id(s_item_cate)
t_cate2id = f_cate2id(t_item_cate)

def convert_cate_id(item2cate, item2id, cate2id):
    id_item_cate = {}
    for item, cate in item2cate.items():
        id_item_cate[item2id[item]] = cate2id[cate]
    return id_item_cate
s_id_itemcate = convert_cate_id(s_item_cate, item1, s_cate2id)
t_id_itemcate = convert_cate_id(t_item_cate, item2, t_cate2id)

with open(out_root+'source_item_cate.json', 'w') as f:
    json.dump(s_id_itemcate, f)
with open(out_root+'target_item_cate.json', 'w') as f:
    json.dump(t_id_itemcate, f)
with open(out_root+'source_cate2id.json', 'w') as f:
    json.dump(s_cate2id, f)
with open(out_root+'target_cate2id.json', 'w') as f:
    json.dump(t_cate2id, f)