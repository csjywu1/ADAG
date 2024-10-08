import pandas as pd

from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse

from tqdm import tqdm

import torch.nn.functional as F


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='ADAG')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=390) #390
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
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
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        # 使用line embedding训练
        # 9/24 只创建一次subgraphs
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

        # args.num_epoch
        for epoch in range(args.num_epoch):
            model.train()

            # 随机打乱所有节点的索引
            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            for batch_idx in range(batch_num):
                optimiser.zero_grad()

                # 9/24 遍历根节点，随机生成两种子图得恶意节点
                # 第一种节点是相似，但并不相连得
                # 第二种是矩阵，

                # 确定当前批次的索引
                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                # 创建标签张量
                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(
                    device)

                # 初始化附加的零张量
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.  # 将最后一行设为1
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                # 使用列表收集当前批次的邻接矩阵和特征
                ba_list = []
                ba_hat_list = []
                bf_list = []

                for i in idx:
                    # 提取当前子图的邻接矩阵和特征
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_adj_hat = adj_hat[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]

                    # 将当前的邻接矩阵和特征添加到列表中
                    ba_list.append(cur_adj)
                    ba_hat_list.append(cur_adj_hat)
                    bf_list.append(cur_feat)

                # 拼接所有的邻接矩阵和特征
                ba = torch.cat(ba_list, dim=0)
                ba_hat = torch.cat(ba_hat_list, dim=0)
                bf = torch.cat(bf_list, dim=0)

                # 拼接附加的零张量
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)

                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)

                # 最后处理特征张量
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

        with tqdm(total=args.auc_test_rounds) as pbar_test:
            pbar_test.set_description('Testing')
            for round in range(args.auc_test_rounds):
                all_idx = list(range(nb_nodes))
                random.shuffle(all_idx)
                subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
                for batch_idx in range(batch_num):
                    optimiser.zero_grad()
                    is_final_batch = (batch_idx == (batch_num - 1))
                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]
                    cur_batch_size = len(idx)
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

                    with torch.no_grad():
                        test_logits_1, test_logits_2, _, _ = model(bf, ba)
                        test_logits_1_hat, test_logits_2_hat, _, _ = model(bf, ba_hat)
                        test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                        test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
                        test_logits_1_hat = torch.sigmoid(torch.squeeze(test_logits_1_hat))
                        test_logits_2_hat = torch.sigmoid(torch.squeeze(test_logits_2_hat))

                        logits_2 = test_logits_1[cur_batch_size:2 * cur_batch_size]  # 提取索引区间 [300:600]
                        logits_3 = test_logits_1[2 * cur_batch_size:3 * cur_batch_size]  # 提取索引区间 [600:900]
                        test_logits_1[cur_batch_size:2 * cur_batch_size] = 0 * logits_2 + 1 * logits_3
                        test_logits_1 = test_logits_1[:2 * cur_batch_size]

                        logits_2_hat = test_logits_1_hat[cur_batch_size:2 * cur_batch_size]  # 提取索引区间 [300:600]
                        logits_3_hat = test_logits_1_hat[2 * cur_batch_size:3 * cur_batch_size]  # 提取索引区间 [600:900]
                        test_logits_1_hat[cur_batch_size:2 * cur_batch_size] = 0 * logits_2_hat + 1 * logits_3_hat
                        test_logits_1_hat = test_logits_1_hat[:2 * cur_batch_size]

                        ano_score_1 = - (test_logits_1[:cur_batch_size] -  test_logits_1[cur_batch_size:2 * cur_batch_size]).cpu().numpy()

                        ano_score_1_hat = - (test_logits_1_hat[:cur_batch_size] - test_logits_1_hat[cur_batch_size:2 * cur_batch_size] ).cpu().numpy()

                        # ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                        #     cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                        # ano_score_2_hat = - (
                        #             test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(
                        #         cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                        ano_score = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)

                    multi_round_ano_score[round, idx] = ano_score

                pbar_test.update(1)

            ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
            auc = roc_auc_score(ano_label, ano_score_final)
            all_auc.append(auc)
            print('Testing AUC:{:.4f}'.format(auc), flush=True)

            # 假设已经有了 ano_label 和 ano_score_final
            # 将它们保存为 NumPy 文件
            np.save('ano_label.npy', ano_label)
            np.save('ano_score_final.npy', ano_score_final)

            # 或者将它们保存为 CSV 文件
            df = pd.DataFrame({
                'ano_label': ano_label,
                'ano_score_final': ano_score_final
            })
            df.to_csv('ano_scores.csv', index=False)

            print("Data saved successfully.")


    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')
