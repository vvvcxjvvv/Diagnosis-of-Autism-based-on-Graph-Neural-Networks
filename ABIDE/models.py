import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math
from config import args
from attention import Attention


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout
        # self.MLP = nn.Sequential(
        #     nn.Linear(out, SMC_2),
        #     nn.LogSoftmax(dim=NC_1)
        # )
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            # nn.ReLU()
        )
        self.attn = Attention(args.embed_dim, out_dim=args.hidden_dim, n_head=args.head, score_function='bi_linear',
                              dropout=args.dropout)




    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        # x, _ = self.attn(x, x)
        # x = torch.squeeze(x)
        x = self.gc2(x, adj)
        # x = self.MLP(x)
        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GAT, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout
        # self.MLP = nn.Sequential(
        #     nn.Linear(out, SMC_2),
        #     nn.LogSoftmax(dim=NC_1)
        # )
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            # nn.ReLU()
        )
        # self.attn = Attention(args.embed_dim, out_dim=args.hidden_dim, n_head=args.head, score_function='bi_linear',
        #                       dropout=args.dropout)
        #epoch:75 acc_max: 0.7398 sen_max: 0.7273 spe_max: 0.7500 f1_max: 0.7377 auc_max: 0.7666


        # self.attn = Attention(args.embed_dim, out_dim=args.hidden_dim, n_head=args.head, score_function='mlp',
        #                       dropout=args.dropout)
        # epoch:35 acc_max: 0.7236 sen_max: 0.7018 spe_max: 0.7424 f1_max: 0.7221 auc_max: 0.7658


        self.attn = Attention(args.embed_dim, out_dim=args.hidden_dim, n_head=args.head, score_function='scaled_dot_product',
                              dropout=args.dropout)
        #epoch:42 acc_max: 0.7480 sen_max: 0.7167 spe_max: 0.7778 f1_max: 0.7474 auc_max: 0.7812

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training = self.training)
        x, _ = self.attn(x, x)
        x = torch.squeeze(x)
        x = self.gc2(x, adj)
        return x

class CAMV_GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(CAMV_GCN, self).__init__()
        self.FC = GCN(nfeat, nhid, out, dropout)
        self.HOFC = GCN(nfeat, nhid, out, dropout)
        self.dropout = dropout

        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            # nn.ReLU()
        )

    # def forward(self, x, adj, x1, adj1):
    def forward(self, x, adj):
        z = self.FC(x, adj)
        # z1 = self.HOFC(x1, adj1)
        # zcom = (z + z1) / SMC_2
        # zcom = self.MLP(zcom)
        # return zcom
        z = self.MLP(z)
        return z


class CAMV_GAT(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(CAMV_GAT, self).__init__()
        self.FC = GAT(nfeat, nhid, out, dropout)
        self.HOFC = GAT(nfeat, nhid, out, dropout)
        self.dropout = dropout

        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            # nn.ReLU()
        )


    def forward(self, x, adj):
        z = self.FC(x, adj)
        z = self.MLP(z)
        return z
