import os
import time
import argparse
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import evaluate
import data_utils

import ipdb

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default='XING',help='choose the dataset')
parser.add_argument("--model", type=str,default="IFL",help="model option")

parser.add_argument("--loss_type", type=str,default="log_loss",help="loss option: 'square_loss' or 'log_loss'")
parser.add_argument("--optimizer",type=str,default="Adam",help="optimizer option: 'Adagrad', 'Adam', 'SGD', 'Momentum'")

parser.add_argument("--activation_function",type=str,default="relu",help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
parser.add_argument("--dropout", default='[0.5, 0.2]',  help="dropout rate for FM and MLP")
parser.add_argument("--batch_size", type=int,  default=1024, help="batch size for training")
parser.add_argument("--epochs", type=int,default=300, help="training epochs")

parser.add_argument("--hidden_factor", type=int,default=32,  help="predictive factors numbers in the model")
parser.add_argument("--layers_u",  default='[64]',  help="size of layers in MLP model")
parser.add_argument('--layers_i',default='[64]', help="size of layers in MLP model")
parser.add_argument('--temp', type=float, default=1, help='temperature of contrastive loss')
parser.add_argument("--topN", default='[10, 20, 50, 100]',  help="the recommended item num")
parser.add_argument("--batch_norm", type=int, default=1,   help="use batch_norm or not. option: {1, 0}")

parser.add_argument('--regs', type=float, default=0, help='coefficient of regularization')
parser.add_argument('--regs_mask',type=float, default=0, help='coefficient of regularization of feature mask')
parser.add_argument('--alpha', type=float, default=0.5, help='coefficient of contrastive loss')
parser.add_argument('--beta', type=float, default=0.1, help='coefficient of variance loss')
parser.add_argument('--env', type=int, default=2, help='number of split environments in training')

parser.add_argument("--pre_train", action='store_true', default=False, help="whether use the pre-train or not")
parser.add_argument("--pre_train_model_path", type=str,default="./models/",help="pre_trained model_path")
parser.add_argument("--log_name", type=str, default="log", help="log name")
parser.add_argument("--out",default=True,help="save model or not")
parser.add_argument("--model_path", type=str,default="./models/",help="saved model path")

parser.add_argument("--gpu", type=str,default="0",help="gpu card ID")
parser.add_argument("--ood_test", default=True, help="whether test ood data during iid training")
parser.add_argument('--inference', action='store_true', help='only inferenece')
parser.add_argument('--ckpt', type=str, default=None, help='pre-trained model for inference')

args = parser.parse_args()
print("args:", args)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

def merge_train_dict(dict1,dict2):
    res_dict = dict1
    for userID, his_item_list in dict2.items():
        if userID not in res_dict:
            res_dict[userID] = his_item_list
        else:
            for his_item in his_item_list:
                if his_item in res_dict[userID]:
                    continue
                res_dict[userID].append(his_item)
    return res_dict
#############################  PREPARE DATASET #########################
start_time = time.time()

if args.dataset.upper() == 'XING':
    num_user = 22668
    num_item = 8756
    num_field = [20,9] # this should be modified according to the dataset, the first entry is the number of user features, and the second is the number of item features.
    num_features = 31780
elif args.dataset.upper() == 'MEITUAN':
    num_user = 2113
    num_item = 7138
    num_field = [2,2]
    num_features = 9261

args.data_path = f"../data/{args.dataset.upper()}/"
train_path = args.data_path + 'IID_train.npy'
valid_path = args.data_path + 'IID_val.npy'
test_path = args.data_path + 'IID_test.npy'
feature_value_path = args.data_path + 'IID_train_values.npy'
user_feat_path = args.data_path + 'all_user_features.npy'
item_feat_path = args.data_path + 'all_item_features.npy'

train_dict, train_dict_org = data_utils.load_file(train_path, num_user)
valid_dict, _ = data_utils.load_file(valid_path,num_user)
test_dict, _ = data_utils.load_file(test_path,num_user)

if args.dataset.upper() == "MEITUAN":
    train_dataset = data_utils.XINGData(train_path, feature_value_path, user_feat_path, item_feat_path, num_user, num_item, num_field[1], num_features)
elif args.dataset.upper() == "XING":
    train_dataset = data_utils.XINGData(train_path, feature_value_path, num_user, num_item, num_field[1], num_features)
train_loader = data.DataLoader(train_dataset, drop_last=True,
            batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, worker_init_fn=seed_worker)

if args.ood_test:
    ood_test_path = args.data_path + 'OOD_test.npy'
    ood_test_dict, _ = data_utils.load_file(ood_test_path,num_user)

valid_grd = [[] for _ in range(num_user)]
test_grd = [[] for _ in range(num_user)]
ood_test_grd = [[] for _ in range(num_user)]
for user_id in valid_dict:
    valid_grd[user_id] = valid_dict[user_id]
for user_id in test_dict:
    test_grd[user_id] = test_dict[user_id]
for user_id in ood_test_dict:
    ood_test_grd[user_id] = ood_test_dict[user_id]


print('data ready. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))

##############################  CREATE MODEL ###########################
if args.model == 'IFL':
    model = model.IFL(num_user, num_item, num_field, num_features, args.hidden_factor, args.hidden_factor, eval(args.layers_u), eval(args.layers_i), args.batch_norm, eval(args.dropout), args.temp, args.alpha, args.beta, args.env)
else:
    raise Exception('model not implemented!')

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
###############################  TRAINING ############################

best_recall, best_epoch = -100, 0
best_test_result = []
best_test_ood_result = None

if args.inference:
    model = torch.load('./models/'+args.ckpt)
    model.cuda()
    model.eval()
    with torch.no_grad():
        test_result = evaluate.fullRanking(model, num_user, train_dataset.user_item_mat, test_grd, \
                                                train_dataset.all_user_features, train_dataset.all_item_features, eval(args.topN), 10000)
        evaluate.print_results(None, None, test_result)

        print('--- OOD performance ---')
        test_result_ood = evaluate.fullRanking(model, num_user, train_dataset.user_item_mat, ood_test_grd, \
                                                train_dataset.all_user_features, train_dataset.all_item_features, eval(args.topN), 10000)
        evaluate.print_results(None, None, test_result_ood)
        os._exit(1)

for epoch in range(args.epochs):
    model.train() # Enable dropout and batch_norm
    start_time = time.time()

    for user_feature, user_feat_values, item_feature, item_feat_values, contrastive_item in train_loader:

        user_feature = user_feature.cuda()
        user_feat_values = user_feat_values.cuda()
        item_feature = item_feature.cuda()
        item_feat_values = item_feat_values.cuda()
        contrastive_item = [i.cuda() for i in contrastive_item] 

        model.zero_grad()

        loss = model(user_feature, user_feat_values, item_feature, item_feat_values, contrastive_item)
        loss = loss + args.regs * model.embeddings.weight.norm() + \
                    args.regs_mask * (model.FEATURE_MASK_I.norm() + model.FEATURE_MASK_U.norm())

        loss.backward()
        optimizer.step()
    print("Training Epoch {:03d} ".format(epoch) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
    if  (epoch+1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_result = None
            valid_result, test_result = evaluate.Ranking(model, train_dataset.train_dict, valid_dict, test_dict,\
                                                            train_dataset.all_user_features, train_dataset.all_user_feature_values,\
                                                                    train_dataset.all_item_features, train_dataset.all_item_feature_values, 10000, eval(args.topN))

            print('---'*18)
            print("Runing Epoch {:03d} ".format(epoch) + 'loss {:.4f}'.format(loss) + " costs " + time.strftime(
                                "%H: %M: %S", time.gmtime(time.time()-start_time)))
            evaluate.print_results(None, valid_result, test_result)
            print('---'*18)

            if valid_result[1][1] > best_recall: # recall@50 for selection
                best_recall, best_epoch = valid_result[1][1], epoch
                best_test_result = test_result
                num_decreases = 0
                if args.ood_test:
                    _, ood_test_result = evaluate.Ranking(model, train_dataset.train_dict, valid_dict, ood_test_dict,\
                                                                    train_dataset.all_user_features, train_dataset.all_user_feature_values,\
                                                                            train_dataset.all_item_features, train_dataset.all_item_feature_values, 80000, eval(args.topN))
                    best_test_ood_result = ood_test_result
                    print('--- OOD performance ---')
                    evaluate.print_results(None, None, ood_test_result)
        
                if args.out:
                    print("------------Best model, saving...------------")
                    if not os.path.exists(args.model_path):
                        os.mkdir(args.model_path)
                    torch.save(model, '{}{}_{}_{}lr_{}hidden_{}bs_{}layeru_{}layeri_{}temp_{}alpha_{}beta_{}env_{}regs_{}regsMask_{}drop_{}bn_{}.pth'.format(
                        args.model_path, args.model, args.dataset, args.lr, args.hidden_factor, args.batch_size, \
                        args.layers_u, args.layers_i, args.temp, args.alpha, args.beta, args.env, args.regs, args.regs_mask, args.dropout, args.batch_norm, args.log_name))

            else:
                if num_decreases > 5:
                    print('-'*18)
                    print('Exiting from training early')
                    break
                else:
                    num_decreases += 1

model = torch.load('{}{}_{}_{}lr_{}hidden_{}bs_{}layeru_{}layeri_{}temp_{}alpha_{}beta_{}env_{}regs_{}regsMask_{}drop_{}bn_{}.pth'.format(
                        args.model_path, args.model, args.dataset, args.lr, args.hidden_factor, args.batch_size, \
                        args.layers_u, args.layers_i, args.temp, args.alpha, args.beta, args.env, args.regs, args.regs_mask, args.dropout, args.batch_norm, args.log_name))
model.eval()
with torch.no_grad():
    _, test_result = evaluate.Ranking(model, train_dataset.train_dict, valid_dict, test_dict,\
                                            train_dataset.all_user_features, train_dataset.all_user_feature_values,\
                                                              train_dataset.all_item_features, train_dataset.all_item_feature_values, 80000, eval(args.topN))
    _, best_test_result_ood = evaluate.Ranking(model, train_dataset.train_dict, valid_dict, ood_test_dict,\
                                                        train_dataset.all_user_features, train_dataset.all_user_feature_values,\
                                                             train_dataset.all_item_features, train_dataset.all_item_feature_values, 80000, eval(args.topN))
print('---'*18)
print("End. Best Epoch {:03d}".format(best_epoch))
evaluate.print_results(None, None, test_result)
print('---'*18)

print('--- OOD performance ---')
evaluate.print_results(None, None, best_test_result_ood)




