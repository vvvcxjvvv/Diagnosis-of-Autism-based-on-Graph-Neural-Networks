import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
# from data import load_data, preprocess_features, preprocess_adj, chebyshev_polynomials
# from model import GCN
from models import CAMV_GCN
# from model_final_dgcn import CAMV_GCN
from config import args
from utils import *
import ABIDEParser_1_HOFC_selected as Reader
from scipy.spatial import distance
from scipy import sparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import os
from sklearn.metrics import f1_score
from GATmodel import GAT, SpGAT

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)



#读取view1、view2的特征，下标和标签
subject_IDs = Reader.get_ids_selected()

index = Reader.get_ids2()

labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')

unique = np.unique(list(sites.values())).tolist()



#制作标签
num_classes = 2
num_nodes = len(subject_IDs)
y_data = np.zeros([num_nodes, num_classes])
y = np.zeros([num_nodes, 1])
site = np.zeros([num_nodes, 1], dtype=np.int)
for i in range(num_nodes):
    y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
    y[i] = int(labels[subject_IDs[i]])-1
    site[i] = unique.index(sites[subject_IDs[i]])



# 特征集合
features = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')
features = np.delete(features, np.where(features == 0)[1], axis=1)
features = features.astype(np.float32)




# 分fold
# skf = StratifiedKFold(n_splits=AD_5)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)
cv_splits = list(skf.split(index, np.squeeze(y)))
train = cv_splits[1][0]
train_ind = train
test = cv_splits[1][1]
test_ind = test
trte = np.append(train, test)
val = test

#获取并组合表型信息
index1 = subject_IDs.astype(str)
# graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID', 'AGE_AT_SCAN'], subject_IDs)
graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
labeled_ind = Reader.site_percentage(train, 1.0, subject_IDs)
x_data = Reader.feature_selection(features, y, labeled_ind, 2000)


graph_feat = graph
graph = graph.astype(int)


# 计算view1的邻接矩阵
distv = distance.pdist(x_data, metric='correlation')
dist = distance.squareform(distv)
sigma = np.mean(dist)
sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
final_graph = graph_feat * sparse_graph
adj = final_graph


#预处理特征与支持向量并组合输入
features = sparse.coo_matrix(x_data).tolil()
features = preprocess_features(features)


supports = chebyshev_polynomials(adj, 3)
index = index.astype(np.int64)
y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(y_data, train, test, val)
device = torch.device('cuda')
y_data = torch.from_numpy(y_data).long().to(device)
y_data_one_hot = y_data
y_data = y_data.argmax(dim=1)
train_label = torch.from_numpy(y_train).long().to(device)
train_label_dlk = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(np.int)).to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label_dlk = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(np.int)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label_dlk = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(np.int)).to(device)

#将view1的特征转换为tensor
i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)



i = torch.from_numpy(supports[0][0]).long().to(device)
v = torch.from_numpy(supports[0][1]).to(device)
i1 = torch.from_numpy(supports[1][0]).long().to(device)
v1 = torch.from_numpy(supports[1][1]).to(device)
i2 = torch.from_numpy(supports[2][0]).long().to(device)
v2 = torch.from_numpy(supports[2][1]).to(device)
i3 = torch.from_numpy(supports[3][0]).long().to(device)
v3 = torch.from_numpy(supports[3][1]).to(device)

support_mid = list()
support = torch.sparse.FloatTensor(i.t(), v, supports[0][2]).float().to(device)
support_mid.append(support)
support1 = torch.sparse.FloatTensor(i1.t(), v1, supports[1][2]).float().to(device)
support_mid.append(support1)
support2 = torch.sparse.FloatTensor(i2.t(), v2, supports[2][2]).float().to(device)
support_mid.append(support2)
support3 = torch.sparse.FloatTensor(i3.t(), v3, supports[3][2]).float().to(device)
support_mid.append(support3)





index = torch.from_numpy(index).to(device)

num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]

#model = CAMV_GCN(feat_dim, 256, 96, 0.3)
model = GAT(feat_dim, 256, 2, 0.6, 0.2, 8)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


