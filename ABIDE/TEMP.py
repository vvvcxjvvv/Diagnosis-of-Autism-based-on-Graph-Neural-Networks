# 读取特征，下标和标签
subject_IDs = Reader.get_ids_selected()
index = Reader.get_ids2()
labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
unique = np.unique(list(sites.values())).tolist()

# 制作标签
num_classes = 2
num_nodes = len(subject_IDs)
y_data = np.zeros([num_nodes, num_classes])
y = np.zeros([num_nodes, 1])
site = np.zeros([num_nodes, 1], dtype=np.int)
for i in range(num_nodes):
    y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
    y[i] = int(labels[subject_IDs[i]]) - 1
    site[i] = unique.index(sites[subject_IDs[i]])

# 特征集合
features = Reader.get_networks(subject_IDs, kind='correlation', atlas_name='ho')
features = np.delete(features, np.where(features == 0)[1], axis=1)
features = features.astype(np.float32)
print("features:")
print(features.shape)  # (613, 5050)

# 获取并组合表型信息
index1 = subject_IDs.astype(str)
graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
labeled_ind = Reader.site_percentage(train, 1.0, subject_IDs)
x_data = Reader.feature_selection(features, y, labeled_ind, 2000)

graph_feat = graph
graph = graph.astype(int)

# 计算邻接矩阵
distv = distance.pdist(x_data, metric='correlation')
dist = distance.squareform(distv)
sigma = np.mean(dist)
sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
final_graph = graph_feat * sparse_graph
adj = final_graph

# 预处理特征与支持向量并组合输入
features = sparse.coo_matrix(x_data).tolil()
features = preprocess_features(features)
supports = chebyshev_polynomials(adj, 4)


