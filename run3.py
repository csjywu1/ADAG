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

    optimiser = torch.optim.Adam(adag_model.parameters(), lr=0.001, weight_decay=0.0)
    b_xent_context = nn.BCEWithLogitsLoss(reduction='none')

    # 将 node_embeddings_tensor 赋值给 features，并确保它不参与训练
    adag_model.features = node_embeddings_tensor.to(device)
    # adag_model.features.requires_grad = False

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


    # def generate_subgraph(root_node, neighbors_dict, max_nodes=5, max_hops=3, restart_prob=0.4):
    #     """ 从 root_node 开始构建子图，使用DFS，确保子图的联通性，并避免重复使用节点作为新的根节点 """
    #     G = nx.Graph()
    #     G.add_node(root_node)
    #
    #     # 使用栈实现DFS，元素是 (当前节点, 当前跳数, 当前路径)
    #     stack = [(root_node, 0, [root_node])]  # 栈的初始值为根节点，初始跳数为0
    #     visited = set([root_node])  # 已经访问的节点，避免重复添加
    #
    #     while stack and len(G.nodes) < max_nodes:
    #         current_node, current_hop, path = stack.pop()  # 从栈中取出一个节点、跳数和路径
    #
    #         # print(f"访问节点: {current_node}, 当前路径: {path}")  # 输出当前访问的节点和路径，调试用
    #
    #         # 检查跳数是否超过最大深度
    #         if current_hop >= max_hops:
    #             continue
    #
    #         # 获取当前节点的邻居
    #         neighbors = neighbors_dict.get(current_node, [])
    #
    #         for neighbor in neighbors:
    #             # 如果邻居节点没有访问过并且不是当前节点
    #             if neighbor not in visited and neighbor != current_node:
    #                 visited.add(neighbor)  # 标记该节点已访问
    #                 G.add_edge(current_node, neighbor)  # 添加边 (current_node, neighbor)
    #
    #                 # 更新路径
    #                 new_path = path + [neighbor]
    #
    #                 # 确保子图的联通性，达到最大节点数则返回
    #                 if len(G.nodes) >= max_nodes:
    #                     return G  # 返回已达到最大节点数的子图
    #
    #                 # 如果没有触发restart条件，则继续深搜
    #                 if np.random.rand() > restart_prob:
    #                     stack.append((neighbor, current_hop + 1, new_path))  # 将邻居节点与新的跳数加入栈中
    #
    #                     # 获取邻居的邻居
    #                     neighbor_neighbors = neighbors_dict.get(neighbor, [])
    #                     for nn in neighbor_neighbors:
    #                         # 确保邻居的邻居没有访问过，且避免跳回原节点或循环
    #                         if nn not in visited and nn != current_node and nn != neighbor and len(G.nodes) < max_nodes:
    #                             stack.append((nn, current_hop + 2, new_path + [nn]))  # 将nn加入栈中
    #                             G.add_edge(neighbor, nn)  # 添加边 (neighbor, nn)
    #
    #                             if len(G.nodes) >= max_nodes:  # 如果子图已经达到最大节点数，则立即返回
    #                                 return G
    #
    #     # 如果子图的节点数量不足 max_nodes，则随机增加一些节点
    #     while len(G.nodes) < max_nodes:
    #         existing_nodes = list(G.nodes)
    #         node_to_add = np.random.choice(existing_nodes)  # 随机选择已有的节点
    #         G.add_node(node_to_add)
    #
    #     return G  # 只返回生成的子图

    def generate_rwr_subgraph(root_node, neighbors_dict, max_nodes=5, max_hops=3, restart_prob=0.4):
        """ 从 root_node 开始构建子图，使用DFS，确保子图的联通性，并避免重复使用节点作为新的根节点 """
        G = nx.Graph()
        G.add_node(root_node)

        # 使用栈实现DFS，元素是 (当前节点, 当前跳数)
        stack = [(root_node, 0)]  # 栈的初始值为根节点，初始跳数为0
        visited = set([root_node])  # 已经访问的节点，避免重复添加

        while stack and len(G.nodes) < max_nodes:
            current_node, current_hop = stack.pop()  # 从栈中取出一个节点和跳数

            # 检查跳数是否超过最大深度
            if current_hop >= max_hops:
                continue

            # 获取当前节点的邻居
            neighbors = neighbors_dict.get(current_node, [])

            for neighbor in neighbors:
                # 如果邻居节点没有访问过并且不是当前节点
                if neighbor not in visited:
                    visited.add(neighbor)  # 标记该节点已访问
                    G.add_edge(current_node, neighbor)  # 添加边 (current_node, neighbor)

                    # 确保子图的联通性，达到最大节点数则返回
                    if len(G.nodes) >= max_nodes:
                        return G  # 返回已达到最大节点数的子图

                    # 如果没有触发restart条件，则继续深搜
                    if np.random.rand() > restart_prob:
                        stack.append((neighbor, current_hop + 1))  # 将邻居节点与新的跳数加入栈中

        # 如果子图的节点数量不足 max_nodes，则随机增加一些节点
        while len(G.nodes) < max_nodes:
            existing_nodes = list(G.nodes)
            node_to_add = np.random.choice(existing_nodes)  # 随机选择已有的节点
            G.add_node(node_to_add)

        return G  # 只返回生成的子图


    def generate_negative_samples(G, root_node, neighbors_dict, structure_embeddings, attribute_embeddings,
                                  max_nodes=10):
        """ 生成两个恶意子图，恶意子图1仅包含恶意节点，恶意子图2使用属性特征生成 """

        # --- 构建恶意子图1，只保留恶意节点 ---
        malicious_node1 = None

        # 获取根节点的结构特征嵌入
        root_structure_embedding = structure_embeddings[root_node].unsqueeze(0)  # 保持张量形状 (1, -1)

        # 计算与根节点的余弦相似度（使用结构特征）
        similarity_scores = F.cosine_similarity(root_structure_embedding, structure_embeddings, dim=1)

        # 排序节点，按照相似度从高到低，排除根节点
        sorted_indices = torch.argsort(similarity_scores, descending=True)
        sorted_indices = sorted_indices[sorted_indices != root_node]

        # 随机选择 top 30 中的一个节点作为恶意节点，先取前 30 个节点
        top_30_indices = sorted_indices[:30]
        random_top_30 = random.sample(list(top_30_indices.cpu().numpy()), len(top_30_indices))

        # 遍历相似度最高的前 30 个节点，选择一个与根节点没有连接的节点作为恶意节点
        for node in random_top_30:
            if node not in G.neighbors(root_node):
                malicious_node1 = node
                break

        # 如果遍历结束后还未找到合适的恶意节点，则随机选择一个与根节点没有连接的节点
        if malicious_node1 is None:
            all_nodes = list(G.nodes)
            random.shuffle(all_nodes)

            for node in all_nodes:
                if node != root_node and node not in G.neighbors(root_node):
                    malicious_node1 = node
                    break

        # 如果仍未找到恶意节点，抛出异常
        if malicious_node1 is None:
            raise ValueError(f"无法找到与根节点 {root_node} 不相连的恶意节点。")

        # 确保恶意节点与根节点未连接
        assert malicious_node1 not in G.neighbors(
            root_node), f"恶意节点 {malicious_node1} 与根节点 {root_node} 不应连接！"

        # 仅返回恶意节点，而不生成图
        malicious_subgraph1 = malicious_node1

        # --- 构建恶意子图2 使用属性特征 ---
        malicious_subgraph2 = G.copy()

        # 获取子图中的所有节点
        subgraph_nodes = list(malicious_subgraph2.nodes)

        # 检查子图中所有可能的边对，并选择一条不存在的边
        added_edge = False

        for _ in range(len(subgraph_nodes) ** 2):
            # 随机选择两个不同的节点
            node1, node2 = random.sample(subgraph_nodes, 2)

            # 检查是否没有边连接 node1 和 node2
            if not malicious_subgraph2.has_edge(node1, node2):
                malicious_subgraph2.add_edge(node1, node2)
                added_edge = True
                break

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
    def generate_reference_scores(k=100, mu=0.0, sigma=1.0):
        sampled_scores = torch.normal(mean=mu, std=sigma, size=(k,))
        reference_score = torch.mean(sampled_scores)
        deviation_std = torch.std(sampled_scores)
        return reference_score, deviation_std

    R_s, delta_s = generate_reference_scores()


    def compute_loss(pool_score, noise_pool_score, root_score, noise_root_score, malicious_score,
                     pool_score1, noise_pool_score1, root_score1, pooled_node_score1, malicious_score1):
        # --- 1. 使用 pool_score, noise_pool_score, root_score, noise_root_score, malicious_score 计算交叉熵损失 ---

        # 定义正样本和恶意样本的标签，假设正样本为 1，恶意样本为 0
        labels = torch.cat([torch.ones_like(pool_score), torch.zeros_like(noise_pool_score),
                            torch.ones_like(root_score), torch.zeros_like(noise_root_score),
                            torch.zeros_like(malicious_score)])

        # 将所有分数合并
        scores = torch.cat([pool_score, noise_pool_score, root_score, noise_root_score, malicious_score])
        scores_1_hat = torch.cat([pool_score1, noise_pool_score1, root_score1, pooled_node_score1, malicious_score1])

        # 使用交叉熵损失
        cross_entropy_loss = b_xent_context(scores, labels)
        cross_entropy_loss1 = b_xent_context(scores_1_hat, labels)

        # --- 打印各部分的损失值（调试用） ---
        print(f"Cross Entropy Loss: {torch.mean(cross_entropy_loss).item()}")  # 取均值

        # contrasting loss
        dev_1 = (scores - R_s) / delta_s
        dev_1_hat = (scores_1_hat - R_s) / delta_s

        m = 1
        contrast_loss_1 = (1 - (1 - labels)) * torch.abs(dev_1) + (1 - labels) * torch.clamp(m - torch.abs(dev_1),
                                                                                             min=0)
        contrast_loss_1_hat = (1 - (1 - labels)) * torch.abs(dev_1_hat) + (1 - labels) * torch.clamp(
            m - torch.abs(dev_1_hat), min=0)

        # 计算对比损失的均值
        contrast_loss_1 = torch.mean(contrast_loss_1)
        contrast_loss_1_hat = torch.mean(contrast_loss_1_hat)

        # 返回总损失
        total_loss = 0.1 *(contrast_loss_1 + contrast_loss_1_hat) + torch.mean(cross_entropy_loss) + torch.mean(
            cross_entropy_loss1)
        return total_loss


    # 设置批次大小
    batch_size = 1000

    # 开始训练过程
    # 开始训练过程
    for epoch in range(10):  # 运行3个epoch
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
                positive_subgraph = generate_rwr_subgraph(root_node, neighbors_dict, max_nodes=5, max_hops=3, restart_prob=0.4)
                malicious_subgraph1, malicious_subgraph2 = generate_negative_samples(
                    positive_subgraph, root_node, neighbors_dict, adag_model.features, adag_model.features
                )

                # 将根节点及对应的子图添加到批次列表中
                root_nodes.append(root_node)
                positive_subgraphs.append(positive_subgraph)
                malicious_subgraphs1.append(malicious_subgraph1)
                malicious_subgraphs2.append(malicious_subgraph2)

            # 前向传播，处理整个批次
            pool_score, noise_pool_score, root_score, noise_root_score, malicious_score, positive_pooled_embeddings = adag_model(
                root_nodes, positive_subgraphs, malicious_subgraphs1
            )

            # 把malicious_subgraphs2的逻辑放到后面
            # 前向传播，处理整个批次
            pool_score1, noise_pool_score1, root_score1, pooled_node_score1, malicious_score1, positive_pooled_embeddings1 = adag_model(
                root_nodes, malicious_subgraphs2, malicious_subgraphs1
            )

            # 重点损失
            # 获取当前批次的实际大小
            current_batch_size = positive_pooled_embeddings.size(0)  # 这确保 batch_size 取自当前批次

            # subgraph-subgraph contrast loss
            # subgraph-subgraph contrast loss
            subgraph_embed = F.normalize(positive_pooled_embeddings, dim=1, p=2)
            subgraph_embed_hat = F.normalize(positive_pooled_embeddings1, dim=1, p=2)

            # 计算相似性矩阵
            sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
            sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
            sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())

            # 设置温度系数
            temperature = 1.0

            # 计算指数相似性矩阵
            sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
            sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
            sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)

            # 动态生成 nega_list，以确保它适配当前批次的大小
            nega_list = np.arange(0, current_batch_size - 1, 1)
            nega_list = np.insert(nega_list, 0, current_batch_size - 1)

            # 检查索引是否在范围内
            sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:,
                                                                                                nega_list]

            # 提取对角线元素
            sim_row_sum = torch.diagonal(sim_row_sum)
            sim_diag = torch.diagonal(sim_matrix_one)

            # 计算对角线元素的指数
            sim_diag_exp = torch.exp(sim_diag / temperature)

            # 计算对比损失 (NCE loss)
            NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
            NCE_loss = torch.mean(NCE_loss)

            # 计算批量损失
            batch_loss = compute_loss(
                pool_score, noise_pool_score, root_score, noise_root_score, malicious_score, pool_score1, noise_pool_score1, root_score1, pooled_node_score1, malicious_score1
            )
            batch_loss =0.1 * NCE_loss +  batch_loss


            # 执行反向传播和优化步骤
            optimiser.zero_grad()
            batch_loss.backward()
            optimiser.step()

            # 累积损失并增加批次数量
            total_loss_epoch += batch_loss.item()
            num_batches += 1

            # 输出当前批次的损失
            print(f"Batch {num_batches} Loss: {batch_loss.item()}")

        # 计算并输出当前 epoch 的平均损失
        average_batch_loss = total_loss_epoch / num_batches
        print(f"Epoch {epoch + 1} Completed with Average Batch Loss: {average_batch_loss}")

    final_scores = []  # 用于存储所有节点的最终预测分数

    # 切换模型到评估模式
    adag_model.eval()

    # 禁用梯度计算
    # 禁用梯度计算
    with torch.no_grad():
        # 初始化存储分数的数组，大小为节点总数
        cumulative_scores = np.zeros(nb_nodes)
        cumulative_rounds = 0  # 记录有效的轮次数

        for round in range(10):  # 假设 num_rounds 是要执行的轮次数
            # 生成打乱的节点索引
            shuffled_indices = np.arange(nb_nodes)
            np.random.shuffle(shuffled_indices)

            for batch_start in range(0, nb_nodes, batch_size):
                # 获取当前批次的节点索引
                batch_indices = shuffled_indices[batch_start:min(batch_start + batch_size, nb_nodes)]
                current_batch_size = len(batch_indices)

                # 初始化批量处理的正样本和恶意样本子图
                positive_subgraphs = []
                malicious_subgraphs1 = []
                root_nodes = []

                # 遍历当前批次中的每个根节点，构建子图
                for root_node in batch_indices:
                    # 构建正样本子图和恶意子图
                    positive_subgraph = generate_rwr_subgraph(root_node, neighbors_dict)
                    malicious_subgraph1, _ = generate_negative_samples(
                        positive_subgraph, root_node, neighbors_dict, adag_model.features, adag_model.features
                    )

                    # 将根节点及对应的子图添加到批次列表中
                    root_nodes.append(root_node)
                    positive_subgraphs.append(positive_subgraph)
                    malicious_subgraphs1.append(malicious_subgraph1)

                # 前向传播，处理当前批次
                pool_score, noise_pool_score, root_score, noise_root_score, malicious_score, _ = adag_model(
                    root_nodes, positive_subgraphs, malicious_subgraphs1
                )

                # 使用 squeeze 去掉 root_score 的多余维度，使其与 all_scores 匹配
                cumulative_scores[batch_indices] += root_score.cpu().numpy().squeeze()[:current_batch_size]

            # 每一轮结束后累加轮次数
            cumulative_rounds += 1

        # 在所有轮次结束后，计算最终的平均异常分数
        final_ano_scores = cumulative_scores / cumulative_rounds

        # 计算 AUC，假设 `ano_label` 是你的标签
        auc = roc_auc_score(1 - ano_label, final_ano_scores)

        # 输出 AUC
        print(f'Testing AUC: {auc:.4f}', flush=True)
