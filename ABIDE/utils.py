import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# from NCE.NCEAverage import NCEAverage
# from NCE.NCECriterion import NCECriterion
# from NCE.NCECriterion import NCESoftmaxLoss
import torch.backends.cudnn as cudnn
# from LinearModel import *
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)




######################################################################33
def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_square_error_dirichlet(out, labels, mask):
    """Softmax cross-entropy loss with masking."""
    alpha = torch.exp(out) + 1.0
    # alpha = tf.pow(NC_1.AD_5, preds) + NC_1.0
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = torch.div(alpha, S)
    loss = torch.square(prob - labels) + prob * (1 - prob) / (S + 1.0)
    loss = torch.sum(loss, dim=1)
    mask = mask.float()
    mask /= mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def masked_sen(out, label, label_one_hot):
    # [node, f]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    sen = 0
    spe = 0
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    T_F = correct.int()
    acc = correct.mean()
    for i in range(len(T_F)):
        if T_F[i] == 1 and label[i] == 0:
            TP += 1
        if T_F[i] == 1 and label[i] == 1:
            TN += 1
        if T_F[i] == 0 and label[i] == 0:
            FP += 1
        if T_F[i] == 0 and label[i] == 1:
            FN += 1
    if (TP + FN) != 0:
        sen = TP / (TP + FN)
    if (TN + FP) != 0:
        spe = TN / (TN + FP)
    auc = roc_auc_score(label_one_hot.data.cpu(), out.data.cpu())

    
    return acc, sen, spe, auc


def masked_acc_fed(out, label):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    acc = correct.mean()
    return acc


def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [SMC_2, 49216]
    v = x._values() # [49216]

    # [SMC_2, 4926] => [49216, SMC_2] => [remained node, SMC_2] => [SMC_2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res


def sample_mask1(l):
    """Create mask."""
    mask = np.zeros(l)
    mask[0:49] = 1
    return np.array(mask, dtype=np.bool)


def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    # train_mask = sample_mask1(labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def get_train_test_masks1(labels, idx_train, idx_test, Y, removed_class):
    idx_train_1 = []
    for i in idx_train:
        i = str(i)
        idx_train_1.append(i)
    idxlist = []

    for i in range(len(idx_train_1)):
        if (len(set(Y[idx_train[i]]) & set(removed_class)) == 0):
            idxlist.append(i)

    X_train_cid_idx = [idx_train_1[i] for i in idxlist]
    Y_train_cid = [Y[idx_train[i]] for i in idxlist]
    X_train_cid_idx1 = [idx_train[i] for i in idxlist]

    train_mask = sample_mask(X_train_cid_idx1, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_test, train_mask, test_mask, X_train_cid_idx, Y_train_cid


def get_class_set(labels):
    # labels [l, [c1, c2, ..]]
    # return：the labeled class set
    mydict = {}
    for y in labels:
        for label in y:
            mydict[int(label)] = 1
    return mydict.keys()


def get_labeled_nodes_label_attribute(nodeids, labellist, features):
    '''# Return np[l, ft]
    '''
    label_attribute = get_label_attributes(nodeids, labellist, features)
    #print('label_attribute', label_attribute)
    res = np.zeros([len(nodeids), features.shape[1]])
    for i, labels in enumerate(labellist):
        c = len(labels)
        #print(nodeids[i], 'label number is', c, labels)
        temp = np.zeros([c, features.shape[1]])
        for ii, label in enumerate(labels):
            temp[ii, :] = label_attribute[int(label)]
        temp = np.mean(temp, axis=0)
        res[i, :] = temp
    #print('get_labeled_nodes_label_attribute', res.shape)
    return res


def get_label_attributes(nodeids, labellist, features):
    '''Suppose a node i is labeled as c, then attribute[c] += node_i_attribute, finally mean(attribute[c])
       Input: nodeids, labellist, features
       Output: label_attribute{}: label -> average_labeled_node_features
    '''
    _, feat_num = features.shape
    labels = get_class_set(labellist)
    label_attribute_nodes = defaultdict(list)
    for nodeid, labels in zip(nodeids, labellist):
        for label in labels:
            label_attribute_nodes[int(label)].append(int(nodeid))
    label_attribute = {}
    for label in label_attribute_nodes.keys():
        nodes = label_attribute_nodes[int(label)]
        selected_features = features[nodes, :]
        label_attribute[int(label)] = np.mean(selected_features, axis=0)
    return label_attribute


# def set_model(args, n_data):
#     contrast = NCEAverage(args.feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, False)
#     # contrast = NCEAverage(feat_dim, n_data, args.nce_k, args.nce_t, args.nce_m, args.softmax)
#     criterion_l = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
#     criterion_ab = NCESoftmaxLoss() if args.softmax else NCECriterion(n_data)
#
#     if torch.cuda.is_available():
#         contrast = contrast.cuda()
#         criterion_ab = criterion_ab.cuda()
#         criterion_l = criterion_l.cuda()
#         cudnn.benchmark = True
#
#     return contrast, criterion_ab, criterion_l


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    # R = torch.eye(dim) - (NC_1 / dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost


def accuracy_AM(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



