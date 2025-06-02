import torch
import torch.nn as nn


class conet_unit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conet_unit, self).__init__()
        self.w_t = nn.Linear(in_dim, out_dim, bias=False)
        self.H = nn.Linear(in_dim, out_dim, bias = False)
        self.w_s = nn.Linear(in_dim, out_dim, bias = False)
        self.relu = nn.ReLU()

    def forward(self, x_s, x_t):
        out_s = self.relu( self.w_s(x_s) + self.H(x_t) )
        out_t = self.relu( self.w_t(x_t) + self.H(x_s) )

        return out_s, out_t

class CoNet(nn.Module):
    def __init__(self, config):
        super(CoNet,self).__init__()
        self.config = config
        self.user_emb = nn.Embedding(self.config['user_num'] + 1, self.config['embed_dim'])
        self.t_itemid_emb = nn.Embedding(self.config['t_item_num'] + 1, self.config['embed_dim'])
        self.t_itemcate_emb = nn.Embedding(self.config['t_cate_num'] + 1, self.config['embed_dim'])
        self.s_itemid_emb = nn.Embedding(self.config['s_item_num'] + 1, self.config['embed_dim'])
        self.s_itemcate_emb = nn.Embedding(self.config['s_cate_num'] + 1, self.config['embed_dim'])
        self.device = config['device']
        self.embed_dim = self.config['embed_dim']
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dims = [30, 64, 32, 16, 8]
        self.cross_modules = nn.ModuleList( [ conet_unit(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1) ] )
        self.s_pred = nn.Linear(8,1)
        self.t_pred = nn.Linear(8,1)
        if config['replace']:
            self.replace_pred = nn.Linear(8,1)

    def forward(self, userid, t_can_id, t_can_cate, s_can_id, s_can_cate):
        ###mapping to embeddings, generating masks
        e_user = self.user_emb(userid)
        e_t_can = torch.cat( [self.t_itemid_emb(t_can_id), self.t_itemcate_emb(t_can_cate)], dim = 1)
        e_s_can = torch.cat( [self.s_itemid_emb(s_can_id), self.s_itemcate_emb(s_can_cate)], dim = 1)
        e_source = torch.cat( [e_user, e_s_can], dim = 1 )
        e_target = torch.cat( [e_user, e_t_can], dim = 1 )
        for i in range( len(self.dims)-1 ):
            e_source, e_target = self.cross_modules[i](e_source, e_target)
        r_s = self.sigmoid( self.s_pred(e_source) ).squeeze()
        r_t = self.sigmoid( self.t_pred(e_target) ).squeeze()
        return r_s, r_t

    def meta_prediction(self, userid, t_can_id, t_can_cate, s_can_id, s_can_cate):
        ###mapping to embeddings, generating masks
        e_user = self.user_emb(userid)
        e_t_can = torch.cat( [self.t_itemid_emb(t_can_id), self.t_itemcate_emb(t_can_cate)], dim = 1)
        e_s_can = torch.cat( [self.s_itemid_emb(s_can_id), self.s_itemcate_emb(s_can_cate)], dim = 1)
        e_source = torch.cat( [e_user, e_s_can], dim = 1 )
        e_target = torch.cat( [e_user, e_t_can], dim = 1 )
        for i in range( len(self.dims)-1 ):
            e_source, e_target = self.cross_modules[i](e_source, e_target)
        r_s = self.sigmoid( self.s_pred(e_source) ).squeeze()
        r_t = self.sigmoid( self.replace_pred(e_target) ).squeeze()
        return r_s, r_t

class conet_unit_target(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(conet_unit_target, self).__init__()
        self.w_t = nn.Linear(in_dim, out_dim, bias=False)
        self.relu = nn.ReLU()

    def forward(self,  x_t):
        out_t = self.relu( self.w_t(x_t))
        return out_t

class CoNet_target(nn.Module):
    def __init__(self, config):
        super(CoNet_target,self).__init__()
        self.config = config
        self.user_emb = nn.Embedding(self.config['user_num'] + 1, self.config['embed_dim'])
        self.t_itemid_emb = nn.Embedding(self.config['t_item_num'] + 1, self.config['embed_dim'])
        self.t_itemcate_emb = nn.Embedding(self.config['t_cate_num'] + 1, self.config['embed_dim'])
        self.device = config['device']
        self.embed_dim = self.config['embed_dim']
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dims = [30, 64, 32, 16, 8]
        self.cross_modules = nn.ModuleList( [ conet_unit_target(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1) ] )
        self.t_pred = nn.Linear(8,1)


    def forward(self, userid, t_can_id, t_can_cate, s_can_id, s_can_cate):
        ###mapping to embeddings, generating masks
        e_user = self.user_emb(userid)
        e_t_can = torch.cat( [self.t_itemid_emb(t_can_id), self.t_itemcate_emb(t_can_cate)], dim = 1)
        e_target = torch.cat( [e_user, e_t_can], dim = 1 )
        for i in range( len(self.dims)-1 ):
            e_target = self.cross_modules[i](e_target)
        r_t = self.sigmoid( self.t_pred(e_target) ).squeeze()
        return r_t



