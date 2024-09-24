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
    adj_hat = adj_hat.toarray()  # 将 adj_hat 转换为密集矩阵

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

    # 初始化模型
    input_dim = ft_size  # 输入特征维度
    hidden_dim = 128  # MLP隐藏层维度
    output_dim = 64  # MLP输出维度


    # 输入节点的feature # 用deepwalk的训练得到初始化embedding得到结构特征
    # 假设DeepWalkWithFeatures类已经定义好了
    # Use from_numpy_array instead of from_numpy_matrix
    G = nx.from_numpy_array(adj[0].cpu().numpy())

    ##使用GNNmodel补全这里的代码
    # 定义 GCN 模型类
    class GCNLayer(nn.Module):
        def __init__(self, in_features, out_features):
            super(GCNLayer, self).__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, x, adj):
            out = torch.spmm(adj, x)  # 图卷积操作
            out = self.linear(out)  # 线性变换
            return out


    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCN, self).__init__()
            self.gcn1 = GCNLayer(input_dim, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, output_dim)

        def forward(self, x, adj, return_hidden=False):
            # 第一层 GCN，输出隐藏层的 embedding
            x = self.gcn1(x, adj)
            x_hidden = F.relu(x)  # 隐藏层输出 (隐层有 128 维度)

            # 如果只想输出隐藏层嵌入
            if return_hidden:
                return x_hidden

            # 第二层 GCN，输出最终分类结果
            x = self.gcn2(x_hidden, adj)
            return x


    # 初始化 GCN 模型
    input_dim = features.shape[1]  # 输入特征的维度
    hidden_dim = 128  # 隐藏层维度
    output_dim = 6  # 输出维度为6，因为我们有6个类别
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gnn_model = GCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)


    # 对邻接矩阵进行归一化
    def normalize_adj(adj):
        adj = adj + torch.eye(adj.size(0)).to(device)  # 增加自环
        degree = adj.sum(1)  # 计算度数
        d_inv_sqrt = degree.pow(-0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


    adj_normalized = normalize_adj(torch.tensor(adj[0].cpu().numpy()).to(device))

    # 定义优化器和损失函数
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失

    # 假设 `labels` 是一个包含每个节点的类别标签的张量，shape 为 (nb_nodes,)
    labels = labels.to(device)  # 确保 labels 张量是长整型，并且位于正确的设备上


    def save_embeddings(embeddings, filename='line_embedding.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f'Embeddings saved to {filename}')


    def load_embeddings(filename='line_embedding.pkl'):
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        print(f'Embeddings loaded from {filename}')
        return embeddings


    embedding_filename = 'line_embedding.pkl'
    if os.path.exists(embedding_filename):
        # 如果文件存在，加载嵌入
        node_embeddings_tensor = load_embeddings(embedding_filename)
    else:
        # 如果文件不存在，生成嵌入并保存
        node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=100, p=0.25, q=4)
        node_embeddings = node2vec.fit(window=10, min_count=1).wv
        # 转换为 PyTorch Tensor
        node_embeddings_tensor = torch.FloatTensor(node_embeddings.vectors)
        save_embeddings(node_embeddings_tensor, embedding_filename)

    adag_model = ADAG(input_dim=node_embeddings_tensor.shape[1], hidden_dim=128, output_dim=128).to(device)

    # 将 node_embeddings_tensor 赋值给 features，并确保它不参与训练
    adag_model.features = node_embeddings_tensor.to(device)
    adag_model.features.requires_grad = False

    # 初始化 embeddings 为与 node_embeddings_tensor 大小相同的可训练参数
    adag_model.embeddings = torch.nn.Parameter(torch.Tensor(node_embeddings_tensor.size()).to(device))
    torch.nn.init.xavier_uniform_(adag_model.embeddings)  # 使用 Xavier 初始化
    # 获取相似性邻居集合
    def get_similarity_neighbors(embeddings, top_k=10):
        """ 获取每个节点的相似性邻居集合 """
        # 确保嵌入是一个 Tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.FloatTensor(embeddings)

        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(norm_embeddings, norm_embeddings.t())

        neighbors_dict = {}
        for i in range(similarity_matrix.shape[0]):
            neighbors = torch.argsort(similarity_matrix[i], descending=True)[1:top_k + 1]  # 排序获取 top_k 个邻居
            neighbors_dict[i] = neighbors.cpu().numpy()  # 转换为 NumPy 数组

        return neighbors_dict


    # 生成邻居字典
    neighbors_dict = get_similarity_neighbors(node_embeddings_tensor, top_k=100)


    def generate_subgraph(root_node, neighbors_dict, max_nodes=7, max_hops=3, restart_prob=0.6):
        """ 为每个根节点构建子图，使用DFS，确保子图的联通性 """
        G = nx.Graph()
        G.add_node(root_node)

        # 使用栈而不是队列来实现DFS
        stack = [(root_node, 0)]  # 元素是 (节点, 当前跳数)
        visited = set([root_node])

        while stack and len(G.nodes) < max_nodes:
            current_node, current_hop = stack.pop()  # 从栈中取出一个节点和当前跳数

            # 检查跳数是否超过最大深度
            if current_hop >= max_hops:
                continue

            # 获取当前节点的邻居
            neighbors = neighbors_dict.get(current_node, [])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    G.add_edge(current_node, neighbor)  # 添加当前节点与邻居节点的边

                    # 确保子图的联通性
                    if len(G.nodes) >= max_nodes:
                        return G  # 立即返回已达到最大节点数的子图

                    # 决定是否继续从这个邻居进行搜索
                    if np.random.rand() > restart_prob:  # 如果没有restart，继续从这个邻居进行搜索
                        stack.append((neighbor, current_hop + 1))  # 将邻居与新的跳数加入栈

                        # 将新邻居的邻居加入到stack中，确保继续拓展，但要确保不会超出 max_nodes
                        neighbor_neighbors = neighbors_dict.get(neighbor, [])
                        for nn in neighbor_neighbors:
                            if nn not in visited and len(G.nodes) < max_nodes:
                                stack.append((nn, current_hop + 2))  # 将nn加入栈，并将跳数+2
                                G.add_edge(neighbor, nn)  # 确保边 (neighbor, nn) 也被加入到子图 G 中

                                if len(G.nodes) >= max_nodes:  # 确保不超出 max_nodes
                                    return G

        # 如果子图的节点数量不足 max_nodes，则可以选择复制已有的邻居或随机增加一些节点
        while len(G.nodes) < max_nodes:
            existing_nodes = list(G.nodes)
            node_to_add = np.random.choice(existing_nodes)  # 随机选择一个已有的节点
            G.add_node(node_to_add)

        return G

    def generate_negative_samples(G, root_node, neighbors_dict, structure_embeddings, attribute_embeddings,
                                  max_nodes=10):
        """ 生成两个恶意子图，恶意子图1使用结构特征，恶意子图2使用属性特征 """

        # --- 构建恶意子图1 使用结构特征 ---
        malicious_subgraph1 = G.copy()

        # 获取根节点的结构特征嵌入
        root_structure_embedding = structure_embeddings[root_node].unsqueeze(0)  # 保持张量形状 (1, -1)

        # 计算与根节点的余弦相似度（使用结构特征）
        similarity_scores = F.cosine_similarity(root_structure_embedding, structure_embeddings, dim=1)

        # 排序节点，按照相似度从高到低，排除根节点
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices = sorted_indices[sorted_indices != root_node]

        # 遍历相似度最高的节点，选择一个与根节点没有连接的节点，作为恶意节点
        malicious_node1 = None
        for node in sorted_indices.cpu().numpy():
            if node not in G.neighbors(root_node):
                malicious_node1 = node
                break

        # 如果找到符合条件的恶意节点，加入到恶意子图1中
        if malicious_node1 is not None:
            malicious_subgraph1.add_node(malicious_node1)
            malicious_subgraph1.add_edge(root_node, malicious_node1)

        # --- 构建恶意子图2 使用属性特征 ---
        # --- 构建恶意子图2，尝试在子图范围内添加一条不存在的边 ---
        malicious_subgraph2 = G.copy()

        # 获取子图中的所有节点
        subgraph_nodes = list(malicious_subgraph2.nodes)

        # 检查子图中所有可能的边对，并选择一条不存在的边
        added_edge = False

        for _ in range(len(subgraph_nodes) ** 2):  # 尝试多次，确保找到一个合适的未连接的边
            # 随机选择两个不同的节点
            node1, node2 = random.sample(subgraph_nodes, 2)

            # 检查是否没有边连接 node1 和 node2
            if not malicious_subgraph2.has_edge(node1, node2):
                malicious_subgraph2.add_edge(node1, node2)
                added_edge = True
                break

        # 如果在所有尝试中都没有找到一个合适的边，使用 malicious_subgraph1 替代 malicious_subgraph2
        if not added_edge:
            malicious_subgraph2 = malicious_subgraph1.copy()

        return malicious_subgraph1, malicious_subgraph2


    # 使用1个epoch
    # 对上面的根节点拆分成不同的batch学习
    # 对positive的图卷积 （恶意子图1），对negative的图也卷积 （恶意子图2）
    # 提取对应的embeddings，因此，每个节点会有本身positive的embeddings （1个）以及negative的embeddings （2个）
    # 用MLP网络对embedding计分
    # 基于positive的embeddings（1个）以及negative的embeddings （2个），建立两个对比损失
    # 从标准正态分布中随机选取100个值
    random_values = np.random.normal(loc=0, scale=1, size=100)

    # 计算生成的target_mean 和 target_std
    target_mean = np.mean(random_values)
    target_std = np.std(random_values)


    # 定义 compute_loss 函数
    # 定义 compute_loss 函数
    # 定义 compute_loss 函数
    def compute_loss(positive_score_structure, positive_score_attribute, malicious_score1, malicious_score2):
        # 将score转换为一维
        positive_score_structure = positive_score_structure.squeeze()
        positive_score_attribute = positive_score_attribute.squeeze()
        malicious_score1 = malicious_score1.squeeze()
        malicious_score2 = malicious_score2.squeeze()

        # 确保转换为一维
        positive_score_structure = positive_score_structure.unsqueeze(
            0) if positive_score_structure.dim() == 0 else positive_score_structure
        positive_score_attribute = positive_score_attribute.unsqueeze(
            0) if positive_score_attribute.dim() == 0 else positive_score_attribute
        malicious_score1 = malicious_score1.unsqueeze(0) if malicious_score1.dim() == 0 else malicious_score1
        malicious_score2 = malicious_score2.unsqueeze(0) if malicious_score2.dim() == 0 else malicious_score2

        # --- 1. Margin-based Loss ---
        # 计算结构特征的 margin loss
        margin_loss1_structure = F.relu(1-(malicious_score1 - positive_score_structure ))
        margin_loss_structure = margin_loss1_structure  # 结构损失

        # 计算属性特征的 margin loss
        margin_loss2_attribute = F.relu(1-(malicious_score2 - positive_score_attribute ))
        margin_loss_attribute = margin_loss2_attribute  # 语义损失

        # 总的 margin loss
        total_margin_loss = margin_loss_structure.mean() + margin_loss_attribute.mean()

        # --- 2. Cross-Entropy Loss ---
        # 定义标签，正样本为1，恶意样本为0
        labels = torch.cat([torch.zeros_like(positive_score_structure), torch.ones_like(malicious_score1)])
        # 将正样本和恶意样本分数合并
        scores = torch.cat([positive_score_structure, malicious_score1])

        # 使用交叉熵损失
        cross_entropy_loss_structure = F.binary_cross_entropy_with_logits(scores, labels)

        # 对属性特征进行相同的处理
        labels_attribute = torch.cat([torch.zeros_like(positive_score_attribute), torch.ones_like(malicious_score2)])
        scores_attribute = torch.cat([positive_score_attribute, malicious_score2])
        cross_entropy_loss_attribute = F.binary_cross_entropy_with_logits(scores_attribute, labels_attribute)

        # 总的交叉熵损失
        total_cross_entropy_loss = cross_entropy_loss_structure + cross_entropy_loss_attribute

        # --- 总损失 ---
        total_loss =  total_cross_entropy_loss

        # total_margin_loss +
        # 输出各个部分的损失值
        print(f"Cross Entropy Loss (Structure): {cross_entropy_loss_structure.item()}")
        print(f"Cross Entropy Loss (Attribute): {cross_entropy_loss_attribute.item()}")
        print(f"Total Cross Entropy Loss: {total_cross_entropy_loss.item()}")
        print(f"Total Margin Loss: {total_margin_loss.item()}")
        print(f"Total Loss: {total_loss.item()}")

        return total_loss


    # 设置批次大小
    batch_size = 64

    # 开始训练过程
    # 开始训练过程
    for epoch in range(1):  # 运行3个epoch
        print(f"Starting Epoch {epoch + 1}")

        # 将节点划分为多个批次
        indices = np.arange(nb_nodes)
        np.random.shuffle(indices)  # 打乱节点顺序以增加随机性

        total_loss_epoch = 0.0  # 用于累积当前 epoch 的所有批次损失
        num_batches = 0  # 记录批次数量

        for batch_start in range(0, nb_nodes, batch_size):
            # 获取当前批次的节点
            batch_indices = indices[batch_start:batch_start + batch_size]

            # 初始化用于批量处理的列表
            positive_subgraphs = []
            malicious_subgraphs1 = []
            malicious_subgraphs2 = []
            root_nodes = []

            # 遍历当前批次中的每个根节点，构建子图
            for root_node in batch_indices:
                # 构建正样本子图和恶意子图
                positive_subgraph = generate_subgraph(root_node, neighbors_dict)
                malicious_subgraph1, malicious_subgraph2 = generate_negative_samples(
                    positive_subgraph, root_node, neighbors_dict, adag_model.embeddings, adag_model.embeddings
                )

                # 将根节点及对应的子图添加到批次列表中
                root_nodes.append(root_node)
                positive_subgraphs.append(positive_subgraph)
                malicious_subgraphs1.append(malicious_subgraph1)
                malicious_subgraphs2.append(malicious_subgraph2)

            # 前向传播，处理整个批次
            positive_outputs_structure, positive_outputs_attribute, malicious_outputs1, malicious_outputs2 = adag_model(
                root_nodes, positive_subgraphs, malicious_subgraphs1, malicious_subgraphs2
            )

            # 计算批量损失
            batch_loss = compute_loss(
                positive_outputs_structure, positive_outputs_attribute, malicious_outputs1, malicious_outputs2
            )

            # 执行反向传播和优化步骤
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # 累积损失并增加批次数量
            total_loss_epoch += batch_loss.item()
            num_batches += 1

            # 输出当前批次的损失
            print(f"Batch {num_batches} Loss: {batch_loss.item()}")

        # 计算并输出当前 epoch 的平均损失
        average_batch_loss = total_loss_epoch / num_batches
        print(f"Epoch {epoch + 1} Completed with Average Batch Loss: {average_batch_loss}")

    # 切换模型到评估模式
    adag_model.eval()

    # 存储所有节点的预测分数
    all_structure_scores = []
    all_attribute_scores = []
    final_scores = []

    # 遍历所有节点进行预测
    # 切换模型到评估模式
    adag_model.eval()

    # 存储所有节点的预测分数
    # 遍历所有节点进行批量预测
    final_scores = []  # 用于存储所有节点的最终预测分数
    batch_size = 64  # 根据GPU或CPU的内存情况，可以调整batch_size

    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for batch_start in range(0, nb_nodes, batch_size):
            # 获取当前批次的节点
            batch_indices = np.arange(batch_start, min(batch_start + batch_size, nb_nodes))

            # 初始化用于批量处理的列表
            positive_subgraphs = []
            root_nodes = []

            # 为每个节点构建正样本子图
            for root_node in batch_indices:
                positive_subgraph = generate_subgraph(root_node, neighbors_dict)
                root_nodes.append(root_node)
                positive_subgraphs.append(positive_subgraph)

            # 使用训练好的 ADAG 模型对当前批次节点进行前向传播，计算结构特征和属性特征的 embedding
            positive_scores_structure, positive_scores_attribute, _, _ = adag_model(
                root_nodes, positive_subgraphs, positive_subgraphs, positive_subgraphs
            )

            # 将预测分数转化为列表
            structure_scores = positive_scores_structure.squeeze().tolist()
            attribute_scores = positive_scores_attribute.squeeze().tolist()

            # 计算组合后的最终分数
            for idx in range(len(structure_scores)):
                combined_score = (structure_scores[idx] + attribute_scores[idx]) / 2
                final_scores.append(combined_score)
                # print(f"Node {root_nodes[idx]}: Structure Score = {structure_scores[idx]}, "
                #       f"Attribute Score = {attribute_scores[idx]}, Combined Score = {combined_score}")

    # 将所有预测分数输出为最终结果
    final_predictions = np.array(final_scores)

    # 获取标签的 numpy 数组
    true_labels = ano_label.squeeze()

    # 计算 AUC
    auc_score = roc_auc_score(true_labels, final_predictions)

    print(f"AUC Score: {auc_score}")

    # 一个是使用MLP网络对每个节点的embedding计分，一个是将这个分数放到前面建立的高斯分布中算。如果大于1，则用margin将它转换为1之间
