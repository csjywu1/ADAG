# 保留并提取数据读取部分的代码
import os
import torch
import argparse
import random
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from torch import cosine_similarity, optim
from torch.optim import optimizer
from node2vec import Node2Vec
from ADAG import ADAG
from DeepWalkWithFeatures import DeepWalkWithFeatures
from utils import *
import torch.nn.functional as F

import pickle

from torch_geometric.nn import GCN
from torch_geometric.utils import dense_to_sparse

from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

# 定义参数
parser = argparse.ArgumentParser(description='AGAG Data Loading')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

if __name__ == '__main__':
    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 读取数据部分
    adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(
        args.dataset)

    # 数据预处理
    features, _ = preprocess_features(features)  # 预处理后的特征
    dgl_graph = adj_to_dgl_graph(adj)

    nb_nodes = features.shape[0]  # 获取节点数量
    ft_size = features.shape[1]  # 特征维度
    nb_classes = labels.shape[1]

    # 确保将 adj 和 adj_hat 转换为密集 NumPy 数组
    adj = adj.toarray()  # 将 adj 从稀疏矩阵转换为密集矩阵
    # 确保 adj_hat 被定义
    adj_hat = normalize_adj(adj)  # 根据您的代码逻辑定义或处理 adj_hat
    adj_hat = (adj_hat + sp.eye(adj.shape[0])).todense()
    # adj_hat = adj_hat.toarray()  # 将 adj_hat 转换为密集矩阵

    # 将数据转换为 PyTorch 格式
    features = torch.FloatTensor(features).to(device)  # 将特征转换为 Tensor
    adj = torch.FloatTensor(adj[np.newaxis]).to(device)  # 将 adj 转换为 PyTorch 张量并增加一个维度
    adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)  # 将 adj_hat 转换为 PyTorch 张量并增加一个维度
    labels = torch.FloatTensor(labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)



    print("Data Loaded Successfully")
    print(f"Number of Nodes: {nb_nodes}, Feature Size: {ft_size}, Number of Classes: {nb_classes}")



    # 输入节点的feature # 用deepwalk的训练得到初始化embedding得到结构特征
    # 假设DeepWalkWithFeatures类已经定义好了
    # Use from_numpy_array instead of from_numpy_matrix
    G = nx.from_numpy_array(adj[0].cpu().numpy())


    # 对邻接矩阵进行归一化
    def normalize_adj(adj):
        adj = adj + torch.eye(adj.size(0)).to(device)  # 增加自环
        degree = adj.sum(1)  # 计算度数
        d_inv_sqrt = degree.pow(-0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


    adj_normalized = normalize_adj(torch.tensor(adj[0].cpu().numpy()).to(device))




    def compute_loss():

        return total_loss


    # 设置批次大小
    batch_size = 300

    # 开始训练过程
    # 开始训练过程
    # Function to generate neighbor samples for a given node
    def sample_neighbors(node, adj_matrix, num_neighbors=30):
        neighbors = np.where(adj_matrix[node] > 0)[0]  # Get connected neighbors
        if len(neighbors) < num_neighbors:
            # Pad with -1 if not enough neighbors
            neighbors = np.pad(neighbors, (0, num_neighbors - len(neighbors)), mode='constant', constant_values=-1)
        else:
            # Randomly select num_neighbors from available neighbors
            neighbors = np.random.choice(neighbors, num_neighbors, replace=False)
        return neighbors


    # Function to select connected and unconnected pairs
    def select_pairs(adj_matrix, num_neighbors=30):
        connected_pairs = []
        unconnected_pairs = []

        for node in range(adj_matrix.shape[0]):
            neighbors = np.where(adj_matrix[node] > 0)[0]  # Connected nodes
            non_neighbors = np.where(adj_matrix[node] == 0)[0]  # Unconnected nodes

            if len(neighbors) > 0:
                b = np.random.choice(neighbors, 1)[0]
                connected_pairs.append((node, b))
            if len(non_neighbors) > 0:
                c = np.random.choice(non_neighbors, 1)[0]
                unconnected_pairs.append((node, c))

        return connected_pairs, unconnected_pairs


    # Generate connected and unconnected pairs
    connected_pairs, unconnected_pairs = select_pairs(adj[0].cpu().numpy())

    adag_model = ADAG(input_dim=features.shape[1], hidden_dim=128, output_dim=64).to(device)

    adag_model.features = features
    features.requires_grad = False

    # Start the training loop
    optimizer = torch.optim.Adam(adag_model.parameters(), lr=0.01, weight_decay=5e-4)

    # Start the training loop
    # Start the training loop
    for epoch in range(1):
        total_loss = 0
        np.random.shuffle(connected_pairs)
        np.random.shuffle(unconnected_pairs)

        for i in range(0, len(connected_pairs), batch_size):
            batch_connected = connected_pairs[i:i + batch_size]
            batch_unconnected = unconnected_pairs[i:i + batch_size]

            # Initialize lists to store batch data
            batch_features_a = []
            batch_features_b = []
            batch_features_c = []
            batch_neighbors_a = []
            batch_neighbors_b = []
            batch_neighbors_c = []

            for (a, b), (_, c) in zip(batch_connected, batch_unconnected):
                a_neighbors = sample_neighbors(a, adj[0].cpu().numpy())
                b_neighbors = sample_neighbors(b, adj[0].cpu().numpy())
                c_neighbors = sample_neighbors(c, adj[0].cpu().numpy())

                # Store indices instead of features
                batch_features_a.append(a)  # Use index a
                batch_features_b.append(b)  # Use index b
                batch_features_c.append(c)  # Use index c

                # Collect neighbors indices
                batch_neighbors_a.append(a_neighbors)
                batch_neighbors_b.append(b_neighbors)
                batch_neighbors_c.append(c_neighbors)

            # Convert batch data to tensors and move to the specified device
            batch_features_a = torch.tensor(batch_features_a, dtype=torch.long).to(device)  # Indices as long tensor
            batch_features_b = torch.tensor(batch_features_b, dtype=torch.long).to(device)  # Indices as long tensor
            batch_features_c = torch.tensor(batch_features_c, dtype=torch.long).to(device)  # Indices as long tensor

            # Convert neighbor indices to tensors
            batch_neighbors_a = [torch.tensor(neighbors, dtype=torch.long).to(device) for neighbors in
                                 batch_neighbors_a]
            batch_neighbors_b = [torch.tensor(neighbors, dtype=torch.long).to(device) for neighbors in
                                 batch_neighbors_b]
            batch_neighbors_c = [torch.tensor(neighbors, dtype=torch.long).to(device) for neighbors in
                                 batch_neighbors_c]

            # Forward pass through the model
            optimizer.zero_grad()
            loss = adag_model(batch_features_a, batch_features_b, batch_features_c,
                              batch_neighbors_a, batch_neighbors_b, batch_neighbors_c)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / (len(connected_pairs) // batch_size)}')

    print("Training completed successfully.")
