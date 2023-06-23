import numpy as np 
import torch
import math
import time
import ipdb

def RMSE(model, model_name, dataloader):
    RMSE = np.array([], dtype=np.float32)
    for features, feature_values, label in dataloader:
        features = features.cuda()
        feature_values = feature_values.cuda()
        label = label.cuda()

        prediction = model(features, feature_values)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())

def pre_ranking(item_feature):
    '''prepare for the ranking: construct item_feature data'''

    features = []
    feature_values = []
    
    for itemID in range(len(item_feature)):
        features.append(np.array(item_feature[itemID][0]))
        feature_values.append(np.array(item_feature[itemID][1], dtype=np.float32))
            
    features = torch.tensor(features).cuda()
    feature_values = torch.tensor(feature_values).cuda()
    
    return features, feature_values

def selected_concat(user_feature, all_item_features, all_item_feature_values, userID, batch_item_idx):
    
    item_num = len(batch_item_idx)
    user_feat = torch.tensor(user_feature[userID][0]).cuda()
    user_feat = user_feat.expand(item_num, -1)
    user_feat_values = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32)).cuda()
    user_feat_values = user_feat_values.expand(item_num, -1)
    
    batch_item_idx = torch.tensor(batch_item_idx).cuda()
    batch_item_features = all_item_features[batch_item_idx] 
    batch_item_feature_values = all_item_feature_values[batch_item_idx] 

    features = torch.cat([user_feat, batch_item_features], 1)
    feature_values = torch.cat([user_feat_values, batch_item_feature_values], 1)

    return features, feature_values


def Ranking(model, train_dict, valid_dict, test_dict, all_user_features, all_user_feature_values, all_item_features, all_item_feature_values,\
            batch_size, topN, return_pred=False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    user_gt_test = []
    user_gt_valid = []
    user_pred = []
    user_pred_valid = []
    user_pred_dict = {}
    user_item_top1k = {}
    
    for userID in test_dict:
        batch_num = len(all_item_features)//batch_size
        item_idx = list(range(len(all_item_features)))
        st, ed = 0, batch_size
        mask = torch.zeros(len(all_item_features)).cuda()
        # portion ood would lead to lose of training users
        try:
            his_items = torch.tensor(train_dict[userID]).cuda()
            mask[his_items] = -999
        except:
            if userID not in train_dict:
                pass
            
        for i in range(batch_num):
            batch_item_idx = torch.LongTensor(item_idx[st: ed])
            user_feats = all_user_features[userID].cuda()
            user_feat_values = all_user_feature_values[userID].cuda()
            item_feats = all_item_features[batch_item_idx].cuda()
            item_feat_values = all_item_feature_values[batch_item_idx].cuda()
            
            prediction = model.predict(user_feats, user_feat_values, item_feats, item_feat_values)
            all_predictions = prediction if i ==0 else torch.cat([all_predictions, prediction], 0)

            st, ed = st + batch_size, ed + batch_size
        
        # prediction for the last batch
        batch_item_idx = item_idx[st:]
        user_feats = all_user_features[userID].cuda()
        user_feat_values = all_user_feature_values[userID].cuda()
        item_feats = all_item_features[batch_item_idx].cuda()
        item_feat_values = all_item_feature_values[batch_item_idx].cuda()
        item_feat_values = all_item_feature_values[batch_item_idx].cuda()
        
        prediction = model.predict(user_feats, user_feat_values, item_feats, item_feat_values)

        all_predictions = prediction if batch_num==0 else torch.cat([all_predictions, prediction], 0)
        all_predictions = all_predictions + mask
        
        user_gt_test.append(test_dict[userID])
        _, indices = torch.topk(all_predictions, topN[-1])
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()
        user_item_top1k[userID] = pred_items
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()
        user_pred.append(pred_items)
        if userID in valid_dict:
            user_gt_valid.append(valid_dict[userID])
            user_pred_valid.append(pred_items)  

    valid_results = computeTopNAccuracy(user_gt_valid, user_pred_valid, topN)
    test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)

    if return_pred: # used in the inference.py
        return valid_results, test_results, user_pred_dict, user_item_top1k

    return valid_results, test_results
    

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                user_length += 1
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        
    return precision, recall, NDCG, MRR


def computeTopNAccuracy_avgUser(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        user_length = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                user_length += 1
        
        precision.append(round(sumForPrecision / user_length, 4))
        recall.append(round(sumForRecall / user_length, 4))
        NDCG.append(round(sumForNdcg / user_length, 4))
        MRR.append(round(sumForMRR / user_length, 4))
        
    return precision, recall, NDCG, MRR


def print_results(train_RMSE, valid_result, test_result):
    """output the evaluation results."""
    if train_RMSE is not None:
        print("[Train]: RMSE: {:.4f}".format(train_RMSE))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))
                            
def subRanking(model, num_user, data, grd, all_user_features, all_user_feature_values, all_item_features, all_item_feature_values, topN, n_item=1000, step=100):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    
    start_index = 0 
    end_index = step

    user_tensor = all_user_features
    all_index_of_rank_list = torch.LongTensor([])
    all_score = torch.tensor([]).cuda()

    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index].cuda() # (2000,20)
        temp_user_tensor = temp_user_tensor.repeat(1,n_item).view(-1,20) #([2000000, 20])
        user_feat_values = all_user_feature_values[start_index:end_index].repeat(1,n_item).view(-1,20).cuda() #(2000,20)

        item_feats = all_item_features[(data[start_index:end_index]).reshape(-1)].reshape(temp_user_tensor.size(0),-1).cuda()
        item_feat_values = all_item_feature_values[(data[start_index:end_index]).reshape(-1)].reshape(temp_user_tensor.size(0),-1).cuda()      

        score_matrix = model.sub_predict(temp_user_tensor, user_feat_values, item_feats, item_feat_values)
        score_matrix = score_matrix.view(-1, n_item)

        all_score = torch.cat((all_score, score_matrix)) if len(all_score)!=0 else score_matrix
        start_index = end_index
        if end_index+step < num_user:
            end_index += step
        else:
            end_index = num_user

    _, index_of_rank_list = torch.topk(all_score, topN[-1])
    all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()), dim=0)

    results = computeTopNAccuracy_avgUser(grd, all_index_of_rank_list, topN)

    return results

# def fullRanking(model, num_user, mask, grd, all_user_features, all_item_features, topN, step=100):
#     """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
#     start_index = 0 
#     end_index   = step
#     user_tensor = torch.LongTensor(range(num_user)).cuda()
#     mask        = torch.from_numpy(mask.A)

#     all_index_of_rank_list = torch.LongTensor([])

#     while end_index <= num_user and start_index < end_index:

#         temp_user_tensor = user_tensor[start_index:end_index]
#         user_feats = 
#         score_matrix = model.predict(user_feats, user_feat_values, item_feats, item_feat_values)
#         score_matrix -= 1e8 * mask[start_index:end_index].cuda()

#         start_index = end_index
#         if end_index+step < num_user:
#             end_index += step
#         else:
#             end_index = num_user
#         _, index_of_rank_list = torch.topk(score_matrix, topN[-1])
#         all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()), dim=0)

#     results = computeTopNAccuracy_avgUser(grd, all_index_of_rank_list, topN)
#     return results