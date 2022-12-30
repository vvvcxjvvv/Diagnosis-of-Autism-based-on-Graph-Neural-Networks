import random

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
# from data import load_data, preprocess_features, preprocess_adj, chebyshev_polynomials
# from model import GCN
from models import CAMV_GCN , CAMV_GAT
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
print("features:")
print(features.shape)#(613, 5050)



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
print("graph_shape:{}".format(graph.shape))#graph_shape:(613, 613)
labeled_ind = Reader.site_percentage(train, 1.0, subject_IDs)
print("labeled_ind:{}".format(labeled_ind))
x_data = Reader.feature_selection(features, y, labeled_ind, 2000)
print("x_data:{}".format(x_data))
print("x_data_shape:{}".format(x_data.shape))

'''
labeled_ind:[612, 325, 327, 328, 329, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 344, 345, 346, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314, 315, 316, 317, 319, 321, 322, 323, 324, 604, 605, 606, 608, 609, 611, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 370, 371, 372, 373, 374, 376, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 397, 398, 399, 400, 401, 402, 403, 404, 406, 407, 408, 410, 411, 412, 414, 415, 416, 417, 418, 420, 421, 423, 424, 425, 427, 429, 430, 431, 433, 434, 436, 437, 438, 439, 440, 441, 442, 445, 447, 448, 451, 452, 453, 454, 456, 457, 458, 459, 460, 463, 464, 465, 466, 468, 469, 471, 472, 473, 474, 475, 477, 479, 480, 491, 49, 50, 53, 54, 55, 56, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 30, 31, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 74, 75, 77, 78, 79, 80, 81, 82, 87, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 493, 494, 496, 498, 499, 500, 501, 502, 503, 504, 505, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 101, 102, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 118, 483, 484, 486, 487, 488, 489, 490, 517, 518, 519, 520, 522, 523, 524, 525, 526, 528, 529, 530, 531, 533, 534, 535, 536, 537, 538, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 560, 561, 562, 563, 565, 566, 567, 568, 569, 571, 572, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 591, 593, 594, 595, 596, 597, 598, 599, 600, 119, 120, 125, 128, 129, 131, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 147, 148, 150, 151, 152, 153, 155, 157, 158, 159, 160, 162, 163, 165, 166, 167, 168, 170, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 209, 210, 212, 213, 214, 218, 219, 220, 221, 222, 224, 225, 226, 227, 228, 229, 230, 232, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 246, 247, 252, 253, 254, 255, 256, 257, 258, 260, 261, 262, 263, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 282, 283, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294]
Number of labeled samples 490
Number of features selected 2000
x_data:[[0.6876087  0.67838854 0.47769424 ... 0.69001645 0.72559834 1.3723036 ]
 [0.6826874  0.4376768  0.20137373 ... 0.23837702 0.15046118 1.2581708 ]
 [0.7848649  0.583613   0.73772866 ... 0.23390199 0.25274763 1.4912679 ]
 ...
 [0.68828654 0.26676527 0.72100353 ... 0.54395074 0.57515895 1.3469284 ]
 [0.6312637  0.49608532 0.4566152  ... 0.25384685 0.21360001 1.5670881 ]
 [0.6668792  0.44262174 0.2687101  ... 0.32801506 0.45719245 1.0768266 ]]
 x_data_shape:(613, 2000)
'''

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


supports = chebyshev_polynomials(adj, 4)
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

model = CAMV_GAT(feat_dim, 256, 96, 0.3)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)



def train(model, epochs):
    model.train()
    out2 = model(feature, support_mid)
    cross_loss = masked_loss(out2, train_label, train_mask)
    loss = cross_loss
    acc, _, _, _ = masked_sen(out2[train_ind], y_data[train_ind], y_data_one_hot[train_ind])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc_test, macro_f1, sen, spe, auc = main_test(model)
    print('e:{}'.format(epochs),
          'ltr: {:.4f}'.format(loss.item()),
          'atr: {:.4f}'.format(acc.item()),
          'ate: {:.4f}'.format(acc_test.item()),
          'f1te:{:.4f}'.format(macro_f1.item()))
    return loss.item(), acc_test.item(), macro_f1.item(), sen, spe, auc


def main_test(model):
    model.eval()
    out2 = model(feature, support_mid)
    acc_test, sen, spe, auc = masked_sen(out2[test_ind], y_data[test_ind], y_data_one_hot[test_ind])
    label_max = []
    for idx in test:
        label_max.append(torch.argmax(out2[idx]).item())
    labelcpu = test_label[test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')
    return acc_test, macro_f1, sen, spe, auc




acc_max = 0
f1_max = 0
sen_max = 0
spe_max = 0
auc_max = 0
epoch_max = 0
for epoch in range(args.epochs):
    loss, acc_test, macro_f1, sen, spe, auc = train(model, epoch)
    if acc_test >= acc_max:
        acc_max = acc_test
        f1_max = macro_f1
        sen_max = sen
        spe_max = spe
        auc_max = auc
        epoch_max = epoch
        torch.save(model, r'F:\demo\GCN\GCN_ASD\ABIDE\model\ASD.pth')
print('epoch:{}'.format(epoch_max),
      'acc_max: {:.4f}'.format(acc_max),
      'sen_max: {:.4f}'.format(sen_max),
      'spe_max: {:.4f}'.format(spe_max),
      'f1_max: {:.4f}'.format(f1_max),
      'auc_max: {:.4f}'.format(auc_max))


