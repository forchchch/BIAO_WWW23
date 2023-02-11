import torch
import json
from torch.utils.data import Dataset
import os
import numpy as np
import random

def pad_data(his, maxlen):
    his_len = len(his)
    if his_len >= maxlen:
        pad_his = his[-maxlen:]
        return pad_his, maxlen
    else:
        pad_his = his + [0]*(maxlen-his_len)
        return pad_his, his_len

def obtain_cate(his, item2cate):
    cate = []
    for item in his:
        if item == 0:
            cate.append(0)
        else:
            cate.append(item2cate[str(item)])
    return cate

class Conet_trainset(Dataset):
    def __init__(self, used_set, s_item2cate, t_item2cate, source_dict, source_behave, target_behave, maxhis=10, maxlen=20):
        self.set = used_set
        self.source_dict = source_dict
        self.s_item2cate = s_item2cate
        self.t_item2cate = t_item2cate
        self.source_behave = source_behave
        self.target_behave = target_behave
        self.maxhis = maxhis
        self.maxlen = maxlen

    def __getitem__(self, index):
        record = self.set[index]
        user, candidate, rate,_,_ = record
        can_cate = self.t_item2cate[str(candidate)]
        s_effect= 1.0
        if len(self.source_dict[str(user)]) == 0:
            s_rate, s_candidate = 0.0, 1
            s_effect = 0.0
        else:
            s_rate, s_candidate = random.choice( self.source_dict[str(user)] )
        s_can_cate = self.s_item2cate[str(s_candidate)]
        if s_rate > 3.0:
            s_click = 1
        else:
            s_click = 0

        if rate>3.0:
            click=1
        else:
            click = 0

        recent_pad_t_his, recent_t_len = pad_data(self.target_behave[str(user)], self.maxlen)
        recent_pad_s_his, recent_s_len = pad_data(self.source_behave[str(user)], self.maxlen)
        recent_t_cate = obtain_cate(recent_pad_t_his, self.t_item2cate)
        recent_s_cate = obtain_cate(recent_pad_s_his, self.s_item2cate)

        return user, click, candidate, can_cate, s_click, s_candidate, s_can_cate,  np.asarray(recent_pad_t_his), np.asarray(recent_t_cate), recent_t_len, np.asarray(recent_pad_s_his), np.asarray(recent_s_cate), recent_s_len, s_effect
        

    def __len__(self):
        return len(self.set)
