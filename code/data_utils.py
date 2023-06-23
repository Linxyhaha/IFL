import numpy as np
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy

def load_file(path, n_user):
    interaction_dict = {}
    interaction_dict_orgID = {}
    file = np.load(path, allow_pickle=True)
    for interaction in file:
        userID, itemID,  = interaction[0], interaction[1]
        if userID not in interaction_dict:
            interaction_dict[userID] = []
            interaction_dict_orgID[userID] = []
        interaction_dict[userID].append(itemID-n_user)
        interaction_dict_orgID[userID].append(itemID)
    res_dict = copy.deepcopy(interaction_dict)
    res_dict_org = copy.deepcopy(interaction_dict_orgID)
    return res_dict, res_dict_org

class MeituanData(data.Dataset):
    def __init__(self, train_path, feat_value_path,user_feature_path, item_feature_path, num_user, num_item, num_field, num_features, mask_method='random'):
        super(MeituanData, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_field = num_field
        self.mask_idx = num_features
        self.mask_method = mask_method

        self.user_item_mat = sp.dok_matrix((self.num_user+1, self.num_item+1), dtype=np.float32)

        self.load_data(train_path, feat_value_path)
        self.load_all_features(user_feature_path, item_feature_path)
        
    def load_all_features(self, user_feature_path, item_feature_path):
        user_features_dict = np.load(user_feature_path, allow_pickle=True).item()
        item_feature_dict = np.load(item_feature_path, allow_pickle=True).item()

        self.all_user_features = [None for _ in range(self.num_user)]
        self.all_item_features = [None for _ in range(self.num_item)]
        try:
            for u_id in user_features_dict:
                self.all_user_features[u_id] = np.array(user_features_dict[u_id])
            for i_id in item_feature_dict:
                self.all_item_features[i_id] = np.array(item_feature_dict[i_id])
        except:
            ipdb.set_trace()

        self.all_user_features = torch.LongTensor(self.all_user_features)
        self.all_item_features = torch.LongTensor(self.all_item_features)

    def load_data(self, path, feat_value_path):
        self.user_features = []
        self.user_feature_values = []
        self.item_features = []
        self.item_feature_values = []

        self.train_dict = {}
        self.all_user_features = [None for _ in range(self.num_user)]
        self.all_user_feature_values = torch.FloatTensor([[1]*2 for _ in range(self.num_user)]) # this should be modified according to the feature indices in pre-processed data

        self.all_item_features = [None for _ in range(self.num_item)]
        self.all_item_feature_values = torch.FloatTensor([[1.0]*2 for _ in range(self.num_item)]) # this should be modified according to the feature indices in pre-processed data

        interactions = np.load(path, allow_pickle=True)
        feat_values = np.load(feat_value_path, allow_pickle=True).tolist()

        for idx,interaction in enumerate(interactions):
            userID, itemID = interaction[0], interaction[1]
            user_feature = [userID] + [interaction[2]] # this should be modified according to the feature indices of pre-processed data
            item_feature = [itemID] + [interaction[3]] # this should be modified 
            user_feat_values = [feat_values[idx][0]] + [feat_values[idx][2]] # this should be modified 
            item_feat_values = [feat_values[idx][1]] + [feat_values[idx][3]] # this should be modified 

            if userID not in self.train_dict:
                self.train_dict[userID] = []
            self.train_dict[userID].append(itemID-self.num_user)

            self.user_features.append(np.array(user_feature))
            self.user_feature_values.append(np.float32(user_feat_values))

            self.item_features.append(np.array(item_feature))
            self.item_feature_values.append(np.float32(item_feat_values))
        
            self.user_item_mat[userID, itemID-self.num_user] = 1


    def __len__(self):
        return len(self.user_features)

    def __getitem__(self,idx):
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        user_feature_values = self.user_feature_values[idx]
        item_feature_values = self.item_feature_values[idx]

        item_pos = random.randint(0,self.num_item-1)
        item_neg = random.randint(0,self.num_item-1)
        while item_neg == item_pos:
            item_neg = random.randint(0,self.num_item-1)

        item_feature_pos = self.all_item_features[item_pos]
        item_feature_neg = self.all_item_features[item_neg]
        
        item_feature_value_pos = self.all_item_feature_values[item_pos].clone()
        item_feature_value_neg = self.all_item_feature_values[item_neg].clone()
        
        return user_feature, user_feature_values, item_feature, item_feature_values, (item_feature_pos, item_feature_value_pos, item_feature_neg, item_feature_value_neg)


class XINGData(data.Dataset):
    def __init__(self, train_path, feat_value_path, num_user, num_item, num_field, num_features, mask_method='random'):
        super(XINGData, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.num_field = num_field
        self.mask_idx = num_features
        self.mask_method = mask_method

        self.user_item_mat = sp.dok_matrix((self.num_user+1, self.num_item+1), dtype=np.float32)
        self.load_data(train_path, feat_value_path)

    def load_data(self, path, feat_value_path):
        self.user_features = []
        self.user_feature_values = []
        self.item_features = []
        self.item_feature_values = []

        self.train_dict = {}
        self.all_user_features = [None for _ in range(self.num_user)]
        self.all_user_feature_values = torch.FloatTensor([[0]*20 for _ in range(self.num_user)])

        self.all_item_features = [None for _ in range(self.num_item)]
        self.all_item_feature_values = torch.FloatTensor([[1.0]*9 for _ in range(self.num_item)])

        interactions = np.load(path, allow_pickle=True)
        feat_values = np.load(feat_value_path, allow_pickle=True).tolist()

        for idx,interaction in enumerate(interactions):
            userID, itemID = interaction[0], interaction[1]
            user_feature = [userID] + interaction[2:21].tolist()
            item_feature = [itemID] + interaction[21:].tolist()
            user_feat_values = [feat_values[idx][0]] + feat_values[idx][2:21]
            item_feat_values = [feat_values[idx][1]] + feat_values[idx][21:]

            if userID not in self.train_dict:
                self.train_dict[userID] = []
            self.train_dict[userID].append(itemID-self.num_user)

            self.user_features.append(np.array(user_feature))
            self.user_feature_values.append(np.float32(user_feat_values))

            self.item_features.append(np.array(item_feature))
            self.item_feature_values.append(np.float32(item_feat_values))
            
            if self.all_item_features[itemID-self.num_user] is None:
                self.all_item_features[itemID-self.num_user] = np.array(item_feature)
            if self.all_user_features[userID] is None:
                self.all_user_features[userID] = np.array(user_feature)
                self.all_user_feature_values[userID] = torch.FloatTensor(user_feat_values)

            self.user_item_mat[userID, itemID-self.num_user] = 1

        self.all_user_features = torch.LongTensor(self.all_user_features)
        self.all_item_features = torch.LongTensor(self.all_item_features)
    
    def __len__(self):
        return len(self.user_features)

    def __getitem__(self,idx):
        user_feature = self.user_features[idx]
        item_feature = self.item_features[idx]
        user_feature_values = self.user_feature_values[idx]
        item_feature_values = self.item_feature_values[idx]

        item_pos = random.randint(0,self.num_item-1)
        item_neg = random.randint(0,self.num_item-1)
        while item_neg == item_pos:
            item_neg = random.randint(0,self.num_item-1)

        item_feature_pos = self.all_item_features[item_pos]
        item_feature_neg = self.all_item_features[item_neg]
        
        item_feature_value_pos = self.all_item_feature_values[item_pos].clone()
        item_feature_value_neg = self.all_item_feature_values[item_neg].clone()
        
        return user_feature, user_feature_values, item_feature, item_feature_values, (item_feature_pos, item_feature_value_pos, item_feature_neg, item_feature_value_neg)
