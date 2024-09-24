from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
from gensim.models import Word2Vec
import networkx as nx
import random

from tqdm import tqdm

import torch.nn.functional as F


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='GRADATE')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default= 1e-3) #1e-3
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64) #64 128
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=500) #390 500
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=5)    # 4
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256) #256
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1, help='how much the first view involves')
parser.add_argument('--beta', type=float, default=0.1, help='how much the second view involves')
args = parser.parse_args()

def generate_reference_scores(k=30, mu=0.0, sigma=1.0):
    sampled_scores = torch.normal(mean=mu, std=sigma, size=(k,))
    reference_score = torch.mean(sampled_scores)
    deviation_std = torch.std(sampled_scores)
    return reference_score, deviation_std

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for run in range(args.runs):

        seed = run + 1
        random.seed(seed)


        batch_size = args.batch_size
        subgraph_size = args.subgraph_size

        adj, features, labels, idx_train, idx_val,\
        idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)



        features, _ = preprocess_features(features)
        dgl_graph = adj_to_dgl_graph(adj)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = labels.shape[1]

        # graph data argumentation
        adj_edge_modification = aug_random_edge(adj, 0.2)
        adj = normalize_adj(adj)
        adj = (adj + sp.eye(adj.shape[0])).todense()
        adj_hat = normalize_adj(adj_edge_modification)
        adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()

        features = torch.FloatTensor(features[np.newaxis]).to(device)
        adj = torch.FloatTensor(adj[np.newaxis]).to(device)
        adj_hat = torch.FloatTensor(adj_hat[np.newaxis]).to(device)
        labels = torch.FloatTensor(labels[np.newaxis]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = torch.LongTensor(idx_val).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)

        all_auc = []


        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout, args).to(device)


        # 预训练嵌入
        print('Pretraining embeddings...')


        def deepwalk_embedding(graph, num_walks=10, walk_length=80, embedding_dim=64, window_size=5, epochs=5):
            # 转换 DGL 图为 NetworkX 图
            G = dgl.to_networkx(graph)

            # 生成随机游走序列
            walks = []
            nodes = list(G.nodes())
            for _ in range(num_walks):
                random.shuffle(nodes)
                for node in nodes:
                    walk = perform_random_walk(G, node, walk_length)
                    walks.append(walk)

            # 使用 Word2Vec 训练嵌入
            model = Word2Vec(walks, vector_size=embedding_dim, window=window_size, min_count=0, sg=1, workers=4,
                             epochs=epochs)
            return model.wv


        def perform_random_walk(G, start_node, walk_length):
            walk = [start_node]
            for _ in range(walk_length - 1):
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    next_node = random.choice(neighbors)
                    walk.append(next_node)
                else:
                    break
            return walk
        # 使用 DeepWalk 进行更快的预训练
        print('Pretraining embeddings with DeepWalk...')
        node_embeddings = deepwalk_embedding(dgl_graph, embedding_dim=args.embedding_dim)

        # 设置最大相似节点数量
        max_similar_nodes = min(2 * args.subgraph_size, nb_nodes)  # 例如，可以设定为子图大小的2倍，确保相似节点不会过多

        # 将生成的嵌入作为模型嵌入初始化
        for node in range(nb_nodes):
            model.embeddings.data[node] = torch.tensor(node_embeddings[str(node)])

        # 使用预训练后的嵌入来生成子图
        node_embeddings = model.embeddings.detach().cpu().numpy()
        subgraphs = []

        # 生成子图的算法
        def random_walk_with_restart(dgl_graph, start_nodes, target_size, restart_prob=0.15, max_attempts=5):
            """
            基于随机游走带重启的方式生成子图，并确保子图是连通的，同时返回子图的邻接矩阵
            """
            for attempt in range(max_attempts):
                visited = set(start_nodes)  # 初始化已访问节点
                subgraph_nodes = list(start_nodes)  # 初始化子图节点
                frontier = list(start_nodes)  # 初始节点集合作为扩展边界

                # 获取整个图的嵌入表示，方便后续计算相似性
                node_embeddings = dgl_graph.ndata['emb'].cpu().numpy()

                while len(visited) < target_size and frontier:
                    if random.random() < restart_prob:  # 以一定概率重启
                        current_node = random.choice(start_nodes)  # 随机选择一个初始节点作为新的起点
                    else:
                        current_node = frontier.pop(0)  # 从边界中选一个节点

                    # 获取当前节点的邻居
                    neighbors = dgl_graph.successors(current_node).numpy()

                    # 根据嵌入相似性对邻居进行排序，优先选择相似度较高的节点
                    neighbor_embeddings = node_embeddings[neighbors]
                    current_embedding = node_embeddings[current_node]

                    # 计算与当前节点的余弦相似度
                    similarities = np.dot(neighbor_embeddings, current_embedding) / (
                                np.linalg.norm(neighbor_embeddings, axis=1) * np.linalg.norm(current_embedding) + 1e-8)
                    sorted_neighbors = [neighbor for _, neighbor in sorted(zip(similarities, neighbors), reverse=True)]

                    for neighbor in sorted_neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            subgraph_nodes.append(neighbor)
                            frontier.append(neighbor)

                        # 达到目标大小后停止扩展
                        if len(visited) >= target_size:
                            break

                # 创建子图并检查连通性
                subgraph = nx.Graph()
                subgraph.add_nodes_from(subgraph_nodes)
                for src in subgraph_nodes:
                    neighbors = dgl_graph.successors(src).numpy()
                    for dst in neighbors:
                        if dst in subgraph_nodes:
                            subgraph.add_edge(src, dst)

                # 检查连通性
                if nx.is_connected(subgraph):
                    # 创建子图的邻接矩阵
                    adj_matrix = np.zeros((len(subgraph_nodes), len(subgraph_nodes)))

                    # 映射原图节点 ID 到子图索引
                    node_to_index = {node: idx for idx, node in enumerate(subgraph_nodes)}

                    for src, dst in subgraph.edges():
                        i, j = node_to_index[src], node_to_index[dst]
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1  # 无向图

                    adj_matrix = torch.FloatTensor(adj_matrix)  # 转换为 PyTorch Tensor

                    return subgraph_nodes, adj_matrix  # 返回子图节点列表和邻接矩阵

                else:
                    print(f"Attempt {attempt + 1}: Subgraph not connected, retrying...")

            # 如果超过最大尝试次数，返回已生成的最大子图
            print("Max attempts reached. Returning possibly disconnected subgraph.")
            return subgraph_nodes, torch.zeros((len(subgraph_nodes), len(subgraph_nodes)))  # 返回可能不连通的子图的邻接矩阵


        subgraphs = []
        adj_matrices = []

        for node in range(nb_nodes):
            similarities = np.dot(node_embeddings, node_embeddings[node])

            # 限制相似节点集合的数量
            most_similar_nodes = np.argsort(similarities)[-max_similar_nodes:]

            # 使用随机游走生成子图和邻接矩阵
            subgraph_nodes, adj_matrix = random_walk_with_restart(dgl_graph, most_similar_nodes, args.subgraph_size)

            subgraphs.append(subgraph_nodes)
            adj_matrices.append(adj_matrix)

        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        # 对每个节点生成embedding
        # 用embedding生成子图
        # 生成负样本
        # 用随机生成的高斯分布作为标准，计算正样本在区间的概率，和负样本在区间的概率
        # 测试中直接用一个round实现对比，分数的计算，然后预测标签
        for epoch in range(args.num_epoch):

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            # subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size) # [N,subgraph_size]

            for batch_idx in range(batch_num): #遍历 batch

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx) # barch_size是300

                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)  # 每个节点有6个负样本

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device) # 每个节点有1个负样本

                ba = []
                ba_hat = []
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits_1, logits_2, subgraph_embed, node_embed = model(bf, ba)
                logits_1_hat, logits_2_hat,  subgraph_embed_hat, node_embed_hat = model(bf, ba_hat)

                #subgraph-subgraph contrast loss
                subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
                subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
                sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
                sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
                sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
                temperature = 1.0
                sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
                sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
                sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
                sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
                sim_row_sum = torch.diagonal(sim_row_sum)
                sim_diag = torch.diagonal(sim_matrix_one)
                sim_diag_exp = torch.exp(sim_diag / temperature)
                NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
                NCE_loss = torch.mean(NCE_loss)

                # logits_1
                logits_2 = logits_1[cur_batch_size:2 * cur_batch_size]  # 提取索引区间 [300:600]
                logits_3 = logits_1[2 * cur_batch_size:3 * cur_batch_size]  # 提取索引区间 [600:900]
                logits_1[cur_batch_size:2 * cur_batch_size] = 0 * logits_2 + 1 * logits_3
                logits_1 = logits_1[:2*cur_batch_size]
                loss_all_1 = b_xent_context(logits_1, lbl_context)
                # loss_all_1_hat
                logits_2_hat = logits_1_hat[cur_batch_size:2 * cur_batch_size]
                logits_3_hat = logits_1_hat[2 * cur_batch_size:3 * cur_batch_size]
                logits_1_hat[2 * cur_batch_size:3 * cur_batch_size] = 0 * logits_2 + 1 * logits_3
                logits_1_hat = logits_1_hat[:2*cur_batch_size]
                loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
                loss_1 = torch.mean(loss_all_1)
                loss_1_hat = torch.mean(loss_all_1_hat)

                # loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                # loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
                # loss_2 = torch.mean(loss_all_2)
                # loss_2_hat = torch.mean(loss_all_2_hat)

                # WU: 加上contrast_loss
                R_s, delta_s = generate_reference_scores()
                dev_1 = (logits_1 - R_s) / delta_s
                dev_1_hat = (logits_1_hat - R_s) / delta_s

                m = 1
                contrast_loss_1 = (1 - lbl_context) * torch.abs(dev_1) + lbl_context * torch.clamp(m - torch.abs(dev_1),
                                                                                                   min=0)

                # Contrast loss for logits_1_hat
                # 使用相同的方法计算logits_1_hat的对比损失
                contrast_loss_1_hat = (1 - lbl_context) * torch.abs(dev_1_hat) + lbl_context * torch.clamp(
                    m - torch.abs(dev_1_hat), min=0)

                # Calculate the mean of the contrast losses
                contrast_loss_1 = torch.mean(contrast_loss_1)
                # contrast_loss_2 = torch.mean(contrast_loss_2)
                contrast_loss_1_hat = torch.mean(contrast_loss_1_hat)

                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat #node-subgraph contrast loss
                # loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat #node-node contrast loss
                loss = loss_1 + 0.1 * NCE_loss + contrast_loss_1 + contrast_loss_1_hat #total loss

                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('Testing AUC!', flush=True)


        # with tqdm(total=args.auc_test_rounds) as pbar_test:
        #     pbar_test.set_description('Testing')
        #     for round in range(args.auc_test_rounds):
        #         all_idx = list(range(nb_nodes))
        #         random.shuffle(all_idx)
        #         subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        #         for batch_idx in range(batch_num):
        #             optimiser.zero_grad()
        #             is_final_batch = (batch_idx == (batch_num - 1))
        #             if not is_final_batch:
        #                 idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        #             else:
        #                 idx = all_idx[batch_idx * batch_size:]
        #             cur_batch_size = len(idx)
        #             ba = []
        #             ba_hat = []
        #             bf = []
        #             added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
        #             added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
        #             added_adj_zero_col[:, -1, :] = 1.
        #             added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
        #             for i in idx:
        #                 cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
        #                 cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
        #                 cur_feat = features[:, subgraphs[i], :]
        #                 ba.append(cur_adj)
        #                 ba_hat.append(cur_adj_hat)
        #                 bf.append(cur_feat)
        #
        #             ba = torch.cat(ba)
        #             ba = torch.cat((ba, added_adj_zero_row), dim=1)
        #             ba = torch.cat((ba, added_adj_zero_col), dim=2)
        #             ba_hat = torch.cat(ba_hat)
        #             ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
        #             ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
        #             bf = torch.cat(bf)
        #             bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)
        #
        #             with torch.no_grad():
        #                 test_logits_1, test_logits_2, _, _ = model(bf, ba)
        #                 test_logits_1_hat, test_logits_2_hat, _, _ = model(bf, ba_hat)
        #                 test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
        #                 test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
        #                 test_logits_1_hat = torch.sigmoid(torch.squeeze(test_logits_1_hat))
        #                 test_logits_2_hat = torch.sigmoid(torch.squeeze(test_logits_2_hat))
        #
        #                 logits_2 = test_logits_1[cur_batch_size:2 * cur_batch_size]  # 提取索引区间 [300:600]
        #                 logits_3 = test_logits_1[2 * cur_batch_size:3 * cur_batch_size]  # 提取索引区间 [600:900]
        #                 test_logits_1[cur_batch_size:2 * cur_batch_size] = 0 * logits_2 + 1 * logits_3
        #                 test_logits_1 = test_logits_1[:2 * cur_batch_size]
        #
        #                 logits_2_hat = test_logits_1_hat[cur_batch_size:2 * cur_batch_size]  # 提取索引区间 [300:600]
        #                 logits_3_hat = test_logits_1_hat[2 * cur_batch_size:3 * cur_batch_size]  # 提取索引区间 [600:900]
        #                 test_logits_1_hat[cur_batch_size:2 * cur_batch_size] = 0 * logits_2_hat + 1 * logits_3_hat
        #                 test_logits_1_hat = test_logits_1_hat[:2 * cur_batch_size]
        #
        #                 ano_score_1 = - (test_logits_1[:cur_batch_size] -  test_logits_1[cur_batch_size:2 * cur_batch_size]).cpu().numpy()
        #
        #                 ano_score_1_hat = - (test_logits_1_hat[:cur_batch_size] - test_logits_1_hat[cur_batch_size:2 * cur_batch_size] ).cpu().numpy()
        #
        #                 # ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
        #                 #     cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
        #                 # ano_score_2_hat = - (
        #                 #             test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(
        #                 #         cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
        #                 ano_score = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)
        #
        #             multi_round_ano_score[round, idx] = ano_score
        #
        #         pbar_test.update(1)
        #
        #     ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
        #     auc = roc_auc_score(ano_label, ano_score_final)
        #     all_auc.append(auc)
        #     print('Testing AUC:{:.4f}'.format(auc), flush=True)


    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
