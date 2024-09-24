import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

class ADAG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ADAG, self).__init__()
        # 定义两个图卷积层
        self.gcn_conv1 = GCNConv(128, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, output_dim)

        self.gcn_conv3 = GCNConv(128, hidden_dim)
        self.gcn_conv4 = GCNConv(hidden_dim, output_dim)

        self.gcn_conv5 = GCNConv(128, hidden_dim)
        self.gcn_conv6 = GCNConv(hidden_dim, output_dim)

        # 定义一个 MLP，用于对卷积后的特征进行打分
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个分数
            nn.Sigmoid()  # 将分数归一化到 [0, 1] 范围内
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个分数
            nn.Sigmoid()  # 将分数归一化到 [0, 1] 范围内
        )

        # 保留唯一的嵌入
        self.embeddings = None  # 唯一的特征嵌入
        self.features = None

        # 定义一个 MLP，用于将嵌入和特征进行融合
        self.feature_embedding_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 输出与 GCN 的输入维度匹配
        )

    def forward(self, root_nodes, positive_subgraphs, malicious_subgraphs1, malicious_subgraphs2):
        # --- 对正样本子图进行图卷积 ---
        positive_root_embeddings = []
        positive_node_embeddings_list = []  # 存储每个正样本子图卷积后的所有节点嵌入
        positive_node_mappings_list = []  # 存储正样本子图的节点映射

        for idx, positive_subgraph in enumerate(positive_subgraphs):
            positive_nodes = list(positive_subgraph.nodes)
            edge_index = torch.tensor(list(positive_subgraph.edges), dtype=torch.long).t().contiguous().cpu()

            # 创建一个节点编号映射，将原始节点编号映射到从 0 开始的连续编号
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(positive_nodes)}

            # 使用这个映射调整 edge_index，使其与 x 的索引对齐
            edge_index = edge_index.apply_(lambda x: node_mapping[x])

            # 转移 edge_index 回到原来的设备
            edge_index = edge_index.to(self.embeddings.device)

            # 提取嵌入和特征，并将它们进行拼接
            x_embed = self.embeddings[positive_nodes]
            x_feat = self.features[positive_nodes]
            x_combined = torch.cat((x_embed, x_feat), dim=1)

            # 将拼接后的结果传入 MLP
            x = self.feature_embedding_mlp(x_combined)

            # 图卷积操作
            x = self.gcn_conv1(x, edge_index)
            x = F.relu(x)
            x = self.gcn_conv2(x, edge_index)

            # 存储卷积后的节点嵌入
            positive_node_embeddings_list.append(x)
            positive_node_mappings_list.append(node_mapping)

            # 获取正样本根节点的嵌入
            positive_root_embedding = x[node_mapping[root_nodes[idx]]]
            positive_root_embeddings.append(positive_root_embedding)

        # 将批量嵌入转化为张量
        positive_root_embeddings = torch.stack(positive_root_embeddings)

        # --- 对恶意子图1和恶意子图2进行图卷积，使用正样本卷积后的嵌入作为初始输入 ---
        def process_subgraphs_with_positive_embedding(root_nodes, subgraphs, conv1, conv2, positive_embeddings_list,
                                                      positive_mappings_list):
            root_embeddings = []
            for idx, subgraph in enumerate(subgraphs):
                nodes = list(subgraph.nodes)
                edge_index = torch.tensor(list(subgraph.edges), dtype=torch.long).t().contiguous().cpu()

                # 创建节点编号映射
                node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(nodes)}

                # 使用映射调整 edge_index
                edge_index = edge_index.apply_(lambda x: node_mapping[x])
                edge_index = edge_index.to(self.embeddings.device)

                # 从正样本卷积后的嵌入中获取对应节点的嵌入
                positive_embedding = positive_embeddings_list[idx]
                positive_mapping = positive_mappings_list[idx]
                x_embed = positive_embedding[[positive_mapping.get(node, 0) for node in nodes]]  # 使用正样本嵌入作为初始节点嵌入

                # 提取特征并将其拼接
                x_feat = self.features[nodes]
                x_combined = torch.cat((x_embed, x_feat), dim=1)

                # 将拼接后的结果传入 MLP
                x = self.feature_embedding_mlp(x_combined)
                x = conv1(x, edge_index)
                x = F.relu(x)
                x = conv2(x, edge_index)

                root_embedding = x[node_mapping[root_nodes[idx]]]
                root_embeddings.append(root_embedding)

            return torch.stack(root_embeddings)

        # 分别处理 malicious_subgraph1 和 malicious_subgraph2，使用正样本卷积后的嵌入
        malicious_root_embeddings1 = process_subgraphs_with_positive_embedding(
            root_nodes, malicious_subgraphs1, self.gcn_conv1, self.gcn_conv2, positive_node_embeddings_list,
            positive_node_mappings_list)
        malicious_root_embeddings2 = process_subgraphs_with_positive_embedding(
            root_nodes, malicious_subgraphs2, self.gcn_conv1, self.gcn_conv2, positive_node_embeddings_list,
            positive_node_mappings_list)

        # 将嵌入传入 MLP 计算分数
        positive_scores = self.mlp(positive_root_embeddings)
        positive_scores1 = self.mlp1(positive_root_embeddings)
        malicious_scores1 = self.mlp(malicious_root_embeddings1)
        malicious_scores2 = self.mlp1(malicious_root_embeddings2)

        return positive_scores, positive_scores1, malicious_scores1, malicious_scores2

# # 初始化ADAG模型时，删除对 embeddings1 的操作
# adag_model = ADAG(input_dim=embeddings.shape[1], hidden_dim=128, output_dim=output_dim).to(device)
#
# # 将 DeepWalk 的嵌入赋值到唯一的 embeddings 变量中
# adag_model.embeddings = nn.Parameter(torch.FloatTensor(embeddings), requires_grad=True).to(device)
#
# # 继续使用相同的 optimizer
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adag_model.parameters()), lr=0.001)
