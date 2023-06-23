import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
from kmeans_pytorch import kmeans

import ipdb
import time


class IFL(nn.Module):
    def __init__(self, num_user, num_item, num_field, num_features, num_factors, dim, layers_user, layers_item, batch_norm, drop_prob, temp, alpha, beta, n_env, sigma=0.5):
        super(IFL, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_field = num_field
        self.num_features = num_features
        self.num_factors = num_factors
        self.dim = dim
        try:
            self.layers_u = layers_user + [dim] if layers_user is not None else None
            self.layers_i = layers_item + [dim] if layers_item is not None else None
        except:
            print(dim)
            print(layers_user)
            print(layers_item)

        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.temp = temp
        self.alpha = alpha
        
        # for invariant loss
        self.beta = beta
        self.n_env = n_env

        self.sigma = sigma

        # for features mask
        feature_mask = torch.zeros(1, self.num_field[1]).normal_(0.5, 0.5)
        feature_mask= torch.clamp(feature_mask,0,1)
        self.FEATURE_MASK_I = torch.nn.Parameter(feature_mask,requires_grad=True)
        self.MASK_NOISE_I = torch.zeros_like(self.FEATURE_MASK_I).normal_(0,1).cuda()

        feature_mask = torch.zeros(1, self.num_field[0]).normal_(0.5, 0.5)
        feature_mask= torch.clamp(feature_mask,0,1)
        self.FEATURE_MASK_U = torch.nn.Parameter(feature_mask,requires_grad=True)
        self.MASK_NOISE_U = torch.zeros_like(self.FEATURE_MASK_U).normal_(0,1).cuda()

        self.embeddings = nn.Embedding(num_features+1, num_factors, padding_idx=num_features)

        DNN_user = []
        DNN_item = []
        if self.batch_norm:
            DNN_user.append(nn.BatchNorm1d(num_factors * num_field[0])) 
            DNN_item.append(nn.BatchNorm1d(num_factors * num_field[1]))       

        DNN_user.append(nn.Dropout(drop_prob[0]))
        DNN_item.append(nn.Dropout(drop_prob[1]))

        # user DNN
        if self.layers_u is not None:
            in_dim = self.num_factors * self.num_field[0]
            for layer in self.layers_u:
                out_dim = layer
                DNN_user.append(nn.Linear(in_dim, out_dim))
                in_dim = out_dim

        # item DNN
        in_dim = self.num_factors * self.num_field[1]
        for layer in self.layers_i:
            out_dim = layer
            DNN_item.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim


        self.DNN_user = nn.Sequential(*DNN_user)
        self.DNN_item = nn.Sequential(*DNN_item)

        self.apply(xavier_normal_initialization)
        xavier_uniform_(self.embeddings.weight, gain=1)

    def split_env(self, user_emb, item_emb, n_env):
        rep = torch.cat((user_emb, item_emb), dim=1)
        cluster_ids, _ = kmeans(X=rep, num_clusters=n_env, distance='euclidean', tqdm_flag=False, device=torch.device('cuda:0'))
        return cluster_ids

    def drop_feature_values(self):
        pos_values = torch.zeros((1,self.num_field[1])).cuda() #self.FEATURE_MASK.clone()
        neg_values = torch.zeros((1,self.num_field[1])).cuda()

        pos_values[self.FEATURE_MASK_I > self.FEATURE_MASK_I.mean()] = 1
        neg_values[self.FEATURE_MASK_I < self.FEATURE_MASK_I.mean()] = 1

        return pos_values, neg_values

    def forward(self, user_feats, user_feat_values, item_feats, item_feat_values_u, contrastive_items):
        # CF supervised
        batch_size = len(user_feats)
        
        user_embed = self.embeddings(user_feats) 

        feat_mask = self.FEATURE_MASK_U + self.sigma * self.MASK_NOISE_U.normal_(0,1) * self.training
        feat_mask = torch.clamp(feat_mask,0,1)
        user_feat_values = feat_mask * user_feat_values

        user_feat_values = user_feat_values.unsqueeze(dim=-1)
        user_embed = user_embed * user_feat_values
        
        item_embed = self.embeddings(item_feats) 

        feat_mask = self.FEATURE_MASK_I + self.sigma * self.MASK_NOISE_I.normal_(0,1) * self.training
        feat_mask = torch.clamp(self.FEATURE_MASK_I,0,1)
        item_feat_values_u = feat_mask * item_feat_values_u

        item_feat_values_u = item_feat_values_u.unsqueeze(dim=-1)
        item_embed = item_embed * item_feat_values_u

        user_embed = user_embed.view(-1, self.num_field[0] * self.num_factors)
        item_embed = item_embed.view(-1, self.num_field[1] * self.num_factors)

        user_rep = self.DNN_user(user_embed)
        item_rep = self.DNN_item(item_embed)
        cluster_ids = self.split_env(user_rep, item_rep, self.n_env)

        samples_loss, main_loss = self.ssl_loss(user_rep, item_rep, self.temp)
        invariant_loss = self.loss_variance(samples_loss, cluster_ids, self.n_env)

        # self-supervised
        item_feat_pos, item_feat_values_pos, item_feat_neg, item_feat_values_neg = contrastive_items
        item_embed_pos = self.embeddings(item_feat_pos)

        pos_values, _ = self.drop_feature_values()

        item_embed_anchor = item_embed_pos * item_feat_values_pos.unsqueeze(dim=-1)
        item_embed_all = item_embed_pos * (item_feat_values_pos * pos_values).unsqueeze(dim=-1)
        

        item_embed_anchor = item_embed_anchor.view(-1, self.num_field[1] * self.num_factors)
        item_embed_all = item_embed_all.view(-1, self.num_field[1] * self.num_factors)

        item_rep1 = self.DNN_item(item_embed_anchor)
        item_rep2 = self.DNN_item(item_embed_all)
        
        _, contrastive_loss = self.ssl_loss(item_rep1, item_rep2, self.temp)

        loss = main_loss + self.alpha * contrastive_loss + self.beta * invariant_loss

        return loss

    def ssl_loss(self, tensor1, tensor2, ssl_temp):
        tensor1 = F.normalize(tensor1, p=2, dim=1)
        tensor2 = F.normalize(tensor2, p=2, dim=1)
        pos_score = torch.sum(tensor1 * tensor2, dim=1)  # [batch_size]
        ttl_score = torch.matmul(tensor1, tensor2.permute(1, 0))

        pos_score = torch.exp(pos_score / ssl_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / ssl_temp), axis=1)  # [batch_size]
        sample_loss = -torch.log(pos_score / ttl_score)
        ssl_loss = torch.mean(sample_loss)
        return sample_loss, ssl_loss

    def contrastive_loss(self, anchor_tensor, all_tensor, temp_value, bs):
        all_score = torch.exp(F.cosine_similarity(anchor_tensor,all_tensor)/temp_value).view(bs, -1)
        pos_score = all_score[:,0]
        all_score = torch.sum(all_score, dim=1)
        loss = (-torch.log(pos_score/all_score)).mean()
        return loss

    def main_loss(self, anchor_tensor, all_tensor, temp_value, bs):
        all_score = torch.exp(F.cosine_similarity(anchor_tensor,all_tensor)/temp_value).view(bs, bs)
        pos_score = torch.diag(all_score) #all_score[:,0]
        all_score = torch.sum(all_score, dim=1)
        loss_samples = (-torch.log(pos_score/all_score))
        loss = loss_samples.mean()
        return loss_samples, loss 

    def predict(self,user_feats, user_feat_values, item_feats, item_feat_values):

        user_embed = self.embeddings(user_feats).unsqueeze(dim=0).repeat(item_feats.size(0), 1, 1)

        feat_mask = torch.clamp(self.FEATURE_MASK_U,0,1)
        user_feat_values = feat_mask * user_feat_values

        user_feat_values = user_feat_values.unsqueeze(dim=0).unsqueeze(dim=-1)
        user_embed = user_embed * user_feat_values

        item_feat_values_u = item_feat_values
        item_embed = self.embeddings(item_feats) 

        feat_mask = torch.clamp(self.FEATURE_MASK_I,0,1)
        item_feat_values_u = feat_mask * item_feat_values_u

        item_feat_values_u = item_feat_values_u.unsqueeze(dim=-1)
        item_embed = item_embed * item_feat_values_u

        user_embed = user_embed.view(-1, self.num_field[0] * self.num_factors)
        item_embed = item_embed.view(-1, self.num_field[1] * self.num_factors)

        user_rep = self.DNN_user(user_embed)
        item_rep = self.DNN_item(item_embed)

        scores = torch.sum(user_rep * item_rep, dim=1)
        return scores

    def loss_variance(self, main_loss, cluster_ids, n_env):
        grad_avg_I, grad_avg_U = 0, 0
        grad_list_I, grad_list_U = [], []
        penalty = 0
        for t in range(n_env):
            for name, param in self.named_parameters():
                if name == 'FEATURE_MASK_I':
                    grad_single_I = grad(main_loss[cluster_ids==t].mean(), param, create_graph=True)[-1].reshape(-1)
                elif name == 'FEATURE_MASK_U':
                    grad_single_U = grad(main_loss[cluster_ids==t].mean(), param, create_graph=True)[-1].reshape(-1)
            grad_avg_I += grad_single_I / n_env
            grad_list_I.append(grad_single_I)
            grad_avg_U += grad_single_U / n_env
            grad_list_U.append(grad_single_U)
        for idx, gradient_I in enumerate(grad_list_I):
            penalty += torch.sum((gradient_I - grad_avg_I)**2)
            penalty += torch.sum((grad_list_U[idx]-grad_avg_U)**2)
        return torch.sum(penalty)


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

def xavier_uniform_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
