from abc import abstractmethod

from torch import nn
from torch.nn.utils import weight_norm
import torch
from torch import nn, einsum
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, d_k=64, dropout=0.1) -> None:
        super().__init__()
        if not self.ifExist(context_dim):
            context_dim = query_dim
        self.heads = heads
        self.scale = d_k ** -0.5
        d_model = d_k * heads
        self.to_q = nn.Linear(query_dim, d_model, bias=False)
        self.to_kv = nn.Linear(context_dim, 2 * d_model, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, query_dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        if not self.ifExist(context):
            context = x
        # print(x.size())
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda mat: rearrange(mat, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        qkT = einsum('b n d, b m d->b n m', q, k) * self.scale
        attention = qkT.softmax(dim=-1)
        attention = einsum('b n m, b m d->b n d', attention, v)
        attention = rearrange(attention, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(attention)

    @staticmethod
    def ifExist(var):
        if var is None:
            return False
        else:
            return True

class DNN(nn.Module):
    def __init__(self,hidden_layers, input_dim):
        super(DNN, self).__init__()
        self.layer_list = []
        for m in range(len(hidden_layers)+1):
            if m == 0:
                self.layer_list.append(nn.Linear(input_dim, hidden_layers[m]))
                self.layer_list.append(nn.ReLU())
            elif m == len(hidden_layers):
                self.layer_list.append( nn.Linear(hidden_layers[m-1],1) )
            else:
                self.layer_list.append( nn.Linear(hidden_layers[m-1], hidden_layers[m]))
                self.layer_list.append( nn.ReLU())
        self.layers = nn.Sequential(*self.layer_list)
    def forward(self, features):
        result = self.layers(features)
        return result

class Select_Interact_reuse_withitem(nn.Module):
    def __init__(self, feature_dim, domain_num,config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.domain_prior = nn.Parameter(torch.ones(1,domain_num))
        self.s_atten = MultiHeadAttention(self.config['embed_dim']*4, self.config['embed_dim']*4, 4, self.config['embed_dim'], 0.0)
        self.t_atten = MultiHeadAttention(self.config['embed_dim']*4, self.config['embed_dim']*4, 4, self.config['embed_dim'], 0.0)
        self.trans = DNN( [32,16], self.config['embed_dim']*8)
        self.item_transform =  nn.Linear(self.config['embed_dim']*2, 1)
        self.nonlinearity = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()
    
    def forward(self, s_hisid, s_hiscate, t_hisid, t_hiscate, s_len, t_len, canid, cancate, ref_model):
        e_t_his = torch.cat( [ref_model.t_itemid_emb(t_hisid), ref_model.t_itemcate_emb(t_hiscate)], dim = 2)
        e_s_his = torch.cat( [ref_model.s_itemid_emb(s_hisid), ref_model.s_itemcate_emb(s_hiscate)], dim = 2)

        s_mask = (torch.arange(self.config['s_hislen'])[None,:].to(self.device)<s_len[:,None]).float() 
        t_mask = (torch.arange(self.config['s_hislen'])[None,:].to(self.device)<t_len[:,None]).float() 
        s_maskr = s_mask.unsqueeze(2)
        t_maskr = t_mask.unsqueeze(2)
        cans = torch.cat( [ref_model.s_itemid_emb(canid), ref_model.s_itemcate_emb(cancate)], dim = 1)
        cans_tile = torch.unsqueeze(cans,1).repeat(1, self.config['s_hislen'], 1)
        e_s_his = torch.cat([e_s_his, cans_tile], dim=2)
        e_t_his = torch.cat([e_t_his, cans_tile], dim=2)
        e_s_his = e_s_his.detach()
        e_t_his = e_t_his.detach()
        s_rep = torch.mean( self.s_atten(e_s_his)*s_maskr, dim=1 )
        t_rep = torch.mean( self.t_atten(e_t_his)*t_maskr, dim=1 )
        # s_rep = torch.sum( self.s_atten(e_s_his)*s_maskr, dim=1 )
        # t_rep = torch.sum( self.t_atten(e_t_his)*t_maskr, dim=1 )        
        item_weight = self.item_transform(cans)
        fuse_feature = torch.cat([s_rep, t_rep], dim=1)
        norm_mask = self.softmax( self.trans(fuse_feature)*item_weight )
        #data_mask = self.nonlinearity(self.domain_prior)*norm_mask
        data_mask = self.relu(self.domain_prior)*norm_mask
        return data_mask

class Select_item(nn.Module):
    def __init__(self, feature_dim, domain_num,config):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.domain_prior = nn.Parameter(torch.ones(1,domain_num))
        self.item_transform = nn.Linear(self.config['embed_dim']*2, 1)
        self.nonlinearity = nn.Softplus()
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, s_hisid, s_hiscate, t_hisid, t_hiscate, s_len, t_len, canid, cancate, ref_model):

        cans = torch.cat( [ref_model.s_itemid_emb(canid), ref_model.s_itemcate_emb(cancate)], dim = 1)
        norm_mask = self.softmax( self.item_transform(cans) )
        data_mask = self.nonlinearity(self.domain_prior)*norm_mask
        return data_mask



