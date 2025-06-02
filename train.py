import torch
import json
import os
from dataset import Conet_trainset
from conet_model import CoNet,CoNet_target
from torch.utils.data import Subset
import time
import argparse
import logging
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import random
from auxilearn.hypernet import Select_Interact_reuse_withitem
from auxilearn.optim import MetaOptimizer

softmax = nn.Softmax(dim=0)
class GradCosine():
    """Implementation of the unweighted version of the alg. in 'Adapting Auxiliary Losses Using Gradient Similarity'

    """

    def __init__(self):
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    @staticmethod
    def _flattening(grad):
        return torch.cat(tuple(g.reshape(-1, ) for i, g in enumerate(grad)), axis=0)

    def get_grad_cos_sim(self, grad1, grad2):
        """Computes cosine simillarity of gradients after flattening of tensors.

        """

        flat_grad1 = self._flattening(grad1)
        flat_grad2 = self._flattening(grad2)

        cosine = nn.CosineSimilarity(dim=0)(flat_grad1, flat_grad2)

        return torch.clamp(cosine, -1, 1)

    def get_grad(self, losses, shared_parameters):
        """

        :param losses: Tensor of losses of shape (n_tasks, )
        :param shared_parameters: model that are not task-specific parameters
        :return:
        """
        main_grad = torch.autograd.grad(losses[0], shared_parameters, retain_graph=True)
        aux_grad = torch.autograd.grad(losses[1], shared_parameters, retain_graph=True)
        if len(losses)==3:
            valid_grad = torch.autograd.grad(losses[2], shared_parameters, retain_graph=True)
        # copy
        grad = tuple(g.clone() for g in main_grad)
        if len(losses)==3:
            cosine = self.get_grad_cos_sim(valid_grad, aux_grad)
        else:
            cosine = self.get_grad_cos_sim(main_grad, aux_grad)
        if cosine > 0:
            grad = tuple(g + ga for g, ga in zip(grad, aux_grad))

        return grad

    def backward(self, losses, shared_parameters, **kwargs):
        shared_grad = self.get_grad(
            losses,
            shared_parameters=shared_parameters
        )
        loss = losses[0]+losses[1]
        loss.backward()
        # update grads for shared weights
        for p, g in zip(shared_parameters, shared_grad):
            p.grad = g

def cal_entropy(w_s):
    w_s = softmax(w_s)
    loss = -torch.sum( (w_s - torch.mean(w_s))*(w_s-torch.mean(w_s)) )
    # loss = (-w_s*torch.log(w_s)).sum()
    return loss
def meta_optimization(metaloader, replace_opt, model, descent_step, crit, replace_scheduler):
    for k,data in enumerate(metaloader):
        if k < descent_step:
            user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data
            label = label.to(device).float()
            user = user.to(device)
            candidate = candidate.to(device)
            can_cate = can_cate.to(device)
            s_label = s_label.to(device).float()
            s_candidate = s_candidate.to(device)
            s_can_cate = s_can_cate.to(device)
            s_result,t_result = model.meta_prediction(user, candidate, can_cate, s_candidate, s_can_cate)
            loss_t = crit(t_result, label).mean()
            replace_opt.zero_grad()
            loss_t.backward()
            replace_opt.step()
        else:
            break
    replace_scheduler.step()
    return


def save_model_and_hyperparameters(model,opt,auxmodel,meta_opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict(),'auxmodel':auxmodel.state_dict(),"meta_opt":meta_opt.meta_optimizer.state_dict()}
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def load_config(init_config):
    with open('./config/conet_params.json', 'r') as f:
        config = json.load(f)
    for key,value in init_config.items():
        config[key] = value
    return config

def obtain_data_stat(config):
    p_user_dict =  "/users.json"
    p_s_item_dict = "/source_item.json"
    p_t_item_dict =  "/target_item.json"
    p_s_cate = "/source_cate2id.json"
    p_t_cate =  "/target_cate2id.json"
    with open(config['in_root']+p_user_dict, 'r') as f:
        user_dict = json.load(f)
    with open(config['in_root']+p_s_item_dict, 'r') as f:
        s_item_dict = json.load(f)
    with open(config['in_root']+p_t_item_dict, 'r') as f:
        t_item_dict = json.load(f)      
    with open(config['in_root']+p_s_cate, 'r') as f:
        s_cate = json.load(f)      
    with open(config['in_root']+p_t_cate, 'r') as f:
        t_cate = json.load(f) 
    user_num = len(user_dict)
    s_item_num = len(s_item_dict)
    t_item_num = len(t_item_dict)
    s_cate_num = len(s_cate)
    t_cate_num = len(t_cate)
    config['user_num'] = user_num
    config['s_item_num'] = s_item_num
    config['t_item_num'] = t_item_num
    config['s_cate_num'] = s_cate_num
    config['t_cate_num'] = t_cate_num
    
def evaluate_model(model, val_loader, device):
    click_true = []
    click_predict = []
    loss_record = []
    model.eval()
    for data in val_loader:
        user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len,_ = data
        label = label.to(device).float()
        user = user.to(device)
        candidate = candidate.to(device)
        can_cate = can_cate.to(device)
        s_label = s_label.to(device).float()
        s_candidate = s_candidate.to(device)
        s_can_cate = s_can_cate.to(device)
        _,t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
        loglosst = criterion(t_result, label)
        loss_record.extend(loglosst.tolist())
        click_true.extend(label.tolist())
        click_predict.extend(t_result.tolist())
    logloss = np.mean(loss_record)
    roc_auc = roc_auc_score(click_true, click_predict)
    model.train()
    return roc_auc, logloss

def evaluate_model_target(model, val_loader, device):
    click_true = []
    click_predict = []
    loss_record = []
    model.eval()
    for data in val_loader:
        user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len,_ = data
        label = label.to(device).float()
        user = user.to(device)
        candidate = candidate.to(device)
        can_cate = can_cate.to(device)
        s_label = s_label.to(device).float()
        s_candidate = s_candidate.to(device)
        s_can_cate = s_can_cate.to(device)
        t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
        loglosst = criterion(t_result, label)
        loss_record.extend(loglosst.tolist())
        click_true.extend(label.tolist())
        click_predict.extend(t_result.tolist())
    logloss = np.mean(loss_record)
    roc_auc = roc_auc_score(click_true, click_predict)
    model.train()
    return roc_auc, logloss

def train_conet_target(model, crit, train_loader, valid_loader, optimizer, config, logger):
    logger.info(f"start training conet target")
    device = config['device']
    model = model.to(device)
    best_auc = 0.50
    best_epoch = -1
    for epoch in range(config['epochs']):
        for (k, data) in enumerate(train_loader):
            model.train()
            user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data
            label = label.to(device).float()
            user = user.to(device)
            candidate = candidate.to(device)
            can_cate = can_cate.to(device)
            s_label = s_label.to(device).float()
            s_candidate = s_candidate.to(device)
            s_can_cate = s_can_cate.to(device)
            s_effect = s_effect.to(device)

            t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
            loss_t = crit(t_result, label).mean()
            total_loss = loss_t
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, training loss: {total_loss.item()}" )
        auc, logloss = evaluate_model_target(model, valid_loader, device)
        if auc > best_auc:
            best_auc = auc
            save_model(model,optimizer)
            best_epoch = epoch
        logger.info(f"epoch:{epoch}, current auc:,{auc:.6f}, current logloss:,{logloss:.6f}, best auc:, {best_auc}, best epoch:{best_epoch}")

def train_conet(model, crit, train_loader, valid_loader, optimizer, config, logger):
    logger.info(f"start training minet")
    device = config['device']
    model = model.to(device)
    best_auc = 0.50
    best_epoch = -1
    for epoch in range(config['epochs']):
        for (k, data) in enumerate(train_loader):
            model.train()
            user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data
            label = label.to(device).float()
            user = user.to(device)
            candidate = candidate.to(device)
            can_cate = can_cate.to(device)
            s_label = s_label.to(device).float()
            s_candidate = s_candidate.to(device)
            s_can_cate = s_can_cate.to(device)
            s_effect = s_effect.to(device)

            s_result,t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
            loss_t = crit(t_result, label).mean()
            loss_s = (crit(s_result, s_label)*s_effect).mean()
            total_loss = loss_t + config['prior_weight']*loss_s
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, training loss: {total_loss.item()}" )
        auc, logloss = evaluate_model(model, valid_loader, device)
        if auc > best_auc:
            best_auc = auc
            save_model(model,optimizer)
            best_epoch = epoch
        logger.info(f"epoch:{epoch}, current auc:,{auc:.6f}, best auc:, {best_auc}, best epoch:{best_epoch}")

def train_conet_select_replace(model, crit, train_loader, valid_loader, optimizer, config, logger, metaloader):
    logger.info(f"start training minet with new designed selection")
    device = config['device']
    model = model.to(device)
    replace_param = [param for name, param in model.named_parameters() if name.startswith('replace')]
    selector = Select_Interact_reuse_withitem(config["embed_dim"]*2, 1, config)
    selector = selector.to(device)
    replace_opt = optim.Adam(replace_param, config['lr'])
    replace_scheduler = optim.lr_scheduler.CosineAnnealingLR(replace_opt, T_max = 1000)
    heads = ['s_pred.weight', 's_pred.bias', 't_pred.weight', 't_pred.bias', 'replace_pred.weight', 'replace_pred.bias']
    model_param = [param for name, param in model.named_parameters() if name not in heads]
    m_optimizer = optim.SGD( selector.parameters(), lr = config['hplr'], momentum = 0.0, weight_decay = config['aux_decay'])
    #m_optimizer = optim.Adam( selector.parameters(), lr = config['hplr'], weight_decay = config['aux_decay'])
    meta_optimizer = MetaOptimizer(meta_optimizer= m_optimizer, hpo_lr = config['convlr'], truncate_iter = 3, max_grad_norm = 10)
    metaloader_iter = iter(metaloader)
    best_auc = 0.50
    best_epoch = -1
    counter = 0
    for epoch in range(config['epochs']):
        for (k, data) in enumerate(train_loader):
            model.train()
            user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data
            label = label.to(device).float()
            user = user.to(device)
            candidate = candidate.to(device)
            can_cate = can_cate.to(device)
            s_label = s_label.to(device).float()
            s_candidate = s_candidate.to(device)
            s_can_cate = s_can_cate.to(device)
            s_effect = s_effect.to(device)

            recent_t_his = recent_t_his.to(device)
            recent_t_cate = recent_t_cate.to(device)
            recent_t_len = recent_t_len.to(device)
            recent_s_his = recent_s_his.to(device)
            recent_s_cate = recent_s_cate.to(device) 
            recent_s_len = recent_s_len.to(device)

            s_result,t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
            loss_t = crit(t_result, label).mean()
            loss_s = (crit(s_result, s_label)*s_effect)
            reshape_loss = loss_s.reshape(-1,1)
            w_s = selector(recent_s_his, recent_s_cate, recent_t_his, recent_t_cate, recent_s_len, recent_t_len, s_candidate, s_can_cate, model)
            total_loss = loss_t + config['prior_weight']*(w_s*reshape_loss).sum()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            counter += 1
            if k%100 == 0:
                logger.info(f"epoch:{epoch},iteration: {k}, target loss: {loss_t.item()}" )
            if k%1000 == 0:
                logger.info(f"learned_weight:{w_s.view(-1)}" )
            if counter%config['interval'] == 0 and epoch>config['pre_epoch']:
                meta_optimization(metaloader, replace_opt, model, config['descent_step'], crit, replace_scheduler)
                try: 
                    user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect  = next(metaloader_iter)
                except StopIteration:
                    metaloader_iter = iter(metaloader)
                    user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data = next(metaloader_iter)
                label = label.to(device).float()
                user = user.to(device)
                candidate = candidate.to(device)
                can_cate = can_cate.to(device)
                s_label = s_label.to(device).float()
                s_candidate = s_candidate.to(device)
                s_can_cate = s_can_cate.to(device)
                s_effect = s_effect.to(device)

                recent_t_his = recent_t_his.to(device)
                recent_t_cate = recent_t_cate.to(device)
                recent_t_len = recent_t_len.to(device)
                recent_s_his = recent_s_his.to(device)
                recent_s_cate = recent_s_cate.to(device) 
                recent_s_len = recent_s_len.to(device)

                s_result,t_result = model.meta_prediction(user, candidate, can_cate, s_candidate, s_can_cate)
                loss_t = crit(t_result, label).mean()
                loss_s = (crit(s_result, s_label)*s_effect)
                reshape_loss = loss_s.reshape(-1,1)
                w_s = selector(recent_s_his, recent_s_cate, recent_t_his, recent_t_cate, recent_s_len, recent_t_len, s_candidate, s_can_cate, model)
                meta_loss = loss_t + 0.0*(w_s*reshape_loss).sum()
                meta_train_loss = 0.0
                for data in train_loader:
                    user, label, candidate, can_cate, s_label, s_candidate, s_can_cate, recent_t_his, recent_t_cate, recent_t_len, recent_s_his, recent_s_cate, recent_s_len, s_effect = data
                    label = label.to(device).float()
                    user = user.to(device)
                    candidate = candidate.to(device)
                    can_cate = can_cate.to(device)
                    s_label = s_label.to(device).float()
                    s_candidate = s_candidate.to(device)
                    s_can_cate = s_can_cate.to(device)
                    s_effect = s_effect.to(device)

                    recent_t_his = recent_t_his.to(device)
                    recent_t_cate = recent_t_cate.to(device)
                    recent_t_len = recent_t_len.to(device)
                    recent_s_his = recent_s_his.to(device)
                    recent_s_cate = recent_s_cate.to(device) 
                    recent_s_len = recent_s_len.to(device)

                    s_result,t_result = model(user, candidate, can_cate, s_candidate, s_can_cate)
                    loss_t = crit(t_result, label).mean()
                    loss_s = (crit(s_result, s_label)*s_effect)
                    reshape_loss = loss_s.reshape(-1,1)
                    w_s = selector(recent_s_his, recent_s_cate, recent_t_his, recent_t_cate, recent_s_len, recent_t_len, s_candidate, s_can_cate, model)
                    entropy_term = config['ent']*cal_entropy(w_s)
                    total_loss = loss_t + config['prior_weight']*(w_s*reshape_loss).sum()                    
                    meta_train_loss += total_loss
                    break
                hyper_grads = meta_optimizer.step(
                    val_loss=meta_loss,
                    train_loss=meta_train_loss,
                    aux_params = list(selector.parameters()),
                    parameters = model_param,
                    return_grads = True,
                    entropy = None
                )
                #print(hyper_grads)
                if counter%500 == 0:
                    logger.info(f"epoch:{epoch} ,iteration:{k}, main loss:{meta_train_loss.item():.6f},meta loss:{meta_loss.item():.6f}")  
                    # logger.info(f"domain prior:{selector.domain_prior}")        

        auc, logloss = evaluate_model(model, valid_loader, device)
        if auc > best_auc:
            best_auc = auc
            save_model_and_hyperparameters(model,optimizer,selector,meta_optimizer)
            best_epoch = epoch
        logger.info(f"epoch:{epoch}, current auc:,{auc:.6f}, best auc:, {best_auc}, best epoch:{best_epoch}")


def save_model(model,opt):
    state = { 'model':model.state_dict(), 'opt':opt.state_dict() }
    #torch.save(state,'./checkponit/best_predictor.pth')
    torch.save(state,os.path.join(path_check,'best_predictor.pth'))
    return

def load_model_test(model,opt,load_dir,test_loader,device):
    info = torch.load(os.path.join(load_dir,'best_predictor.pth'))
    model.load_state_dict(info['model'])
    opt.load_state_dict(info['opt'])
    auc,logloss = evaluate_model(model, test_loader, device)
    return auc,logloss

def load_model_test_target(model,opt,load_dir,test_loader,device):
    info = torch.load(os.path.join(load_dir,'best_predictor.pth'))
    model.load_state_dict(info['model'])
    opt.load_state_dict(info['opt'])
    auc,logloss = evaluate_model_target(model, test_loader, device)
    return auc,logloss

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default="debug", help='experiment name')
args = parser.parse_args()
init_config = vars(args)
config = load_config(init_config)
obtain_data_stat(config)

###### make logger
dataname = config['dataname']


###### load data
in_root = config['in_root']
train_path = 'train_set.json'
valid_path = 'valid_set.json'
test_path = 'test_set.json'
s_item2cate = 'source_item_cate.json'
t_item2cate = 'target_item_cate.json'
source_train_path = 'source_training.json'
source_behavior_path = 'source_user_behavior.json'
target_behavior_path = 'target_user_behavior.json'
source_pos_neg_path = 'source_pos_neg.json'
with open(os.path.join(in_root, train_path),'r') as f:
    train_list = json.load(f)
with open(os.path.join(in_root, valid_path),'r') as f:
    valid_list = json.load(f)
with open(os.path.join(in_root, test_path),'r') as f:
    test_list = json.load(f)
with open(os.path.join(in_root, s_item2cate),'r') as f:
    sour_i2c = json.load(f)
with open(os.path.join(in_root, t_item2cate),'r') as f:
    tar_i2c = json.load(f)
with open(os.path.join(in_root, source_train_path),'r') as f:
    source_train = json.load(f)

with open(os.path.join(in_root, source_behavior_path),'r') as f:
    source_behavior = json.load(f)
with open(os.path.join(in_root, target_behavior_path),'r') as f:
    target_behavior = json.load(f)
with open(os.path.join(in_root, source_pos_neg_path),'r') as f:
    source_pos_neg = json.load(f)
###### load config and data stastics

path_log = os.path.join('./conet_record',dataname,'log',config['exp_name'])
path_check = os.path.join('./conet_record',dataname,'checkpoint',config['exp_name'])

if not os.path.exists(path_log):
    os.makedirs(path_log)
if not os.path.exists(path_check):
    os.makedirs(path_check)

logger = get_logger(os.path.join(path_log, 'logging.txt'))
###### model and dataloader maker
set_seed(config['seed'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
config['device'] = device
trainset = Conet_trainset(train_list, sour_i2c, tar_i2c, source_pos_neg, source_behavior, target_behavior, config['hislen'], config['s_hislen'])
logger.info(f"training set number:{len(trainset)}")
ratio = config['ratio']
meta_val_size = int(len(trainset)*ratio)
if config["use_meta"] == 1:
    with open(os.path.join(in_root, 'meta_indices.json'),'r') as f:
        meta_indice = json.load(f)
    with open(os.path.join(in_root, 'meta_train_indices.json'),'r') as f:
        train_indice = json.load(f)
    # metaset = Subset(trainset, meta_indice)
    # trainset = Subset(trainset, train_indice)
    metaset = trainset
    # trainset, metaset = torch.utils.data.random_split(
    #     trainset, (len(trainset) - meta_val_size, meta_val_size)
    # )
trainloader = DataLoader(trainset, shuffle=True, num_workers=1, batch_size = config["t_batchsize"])
validset = Conet_trainset(valid_list, sour_i2c, tar_i2c, source_pos_neg, source_behavior, target_behavior, config['hislen'], config['s_hislen'])
validloader = DataLoader(validset, shuffle=False, num_workers=1, batch_size = config["test_batchsize"])
testset = Conet_trainset(test_list, sour_i2c, tar_i2c, source_pos_neg, source_behavior, target_behavior, config['hislen'], config['s_hislen'])
testloader = DataLoader(testset, shuffle=False, num_workers=1, batch_size = config["test_batchsize"])
if config["use_meta"] == 1:
    metaloader = DataLoader(metaset, shuffle=False, num_workers=1, batch_size = config["meta_batchsize"])

logger.info(config)
if config["only_target"]:
    conet_model = CoNet_target(config).to(device)
else:
    conet_model = CoNet(config).to(device)
logger.info(conet_model)
criterion = nn.BCELoss(reduction = 'none')
optimizer = optim.Adam(conet_model.parameters(), config['lr'])
if config["only_target"]:
    train_conet_target(conet_model, criterion, trainloader, validloader, optimizer, config, logger)
    auc,logloss = load_model_test_target(conet_model,optimizer,path_check,testloader,device)
else:
    if config["use_meta"] == 0:
        train_conet(conet_model, criterion, trainloader, validloader, optimizer, config, logger)
    else:
        logger.info(f"here we enter the training with bi-level selection")
        train_conet_select_replace(conet_model, criterion, trainloader, validloader, optimizer, config, logger, metaloader)
    auc,logloss = load_model_test(conet_model,optimizer,path_check,testloader,device)
logger.info( f"auc: {auc:.6f}" )



