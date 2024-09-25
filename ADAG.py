import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x.clone())


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)  # Linear transformation of node embeddings
        # if sparse:
        #     out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        # else:
        #     out = torch.bmm(adj, seq_fts)
        #
        # if self.bias is not None:
        #     out += self.bias

        return self.act(seq_fts)


class ADAG(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ADAG, self).__init__()
        # 定义两个图卷积层
        self.gcn_conv1 = GCN(64, hidden_dim)
        self.gcn_conv2 = GCN(hidden_dim, output_dim)

        self.gcn_conv3 = GCN(64, hidden_dim)
        self.gcn_conv4 = GCN(hidden_dim, output_dim)

        self.gcn_conv5 = GCN(64, hidden_dim)
        self.gcn_conv6 = GCN(hidden_dim, output_dim)

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
            nn.Linear(1433, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)  # 输出与 GCN 的输入维度匹配
        )

        self.single_noise = nn.Parameter(torch.randn(1, 64))  # [1, 64]

        self.single_noise1 = nn.Parameter(torch.randn(1, 64))  # [1, 64]


        self.virtual_node = nn.Parameter(torch.randn(1, 64))  # [1, 64]

        self.virtual_node1 = nn.Parameter(torch.randn(1, 64))  # [1, 64]


        self.mlp_combine = SimpleMLP(input_dim=2 * 64, hidden_dim=64,
                                     output_dim=64)

        self.mlp_noise = MLP(64, 64)

        self.mlp_noise1 = MLP(64, 64)


        # Initialize the layers
        self.bilinear_pool1 = nn.Bilinear(64, 64, 1)
        self.bilinear_pool2 = nn.Bilinear(64, 64, 1)
        self.bilinear_pool3 = nn.Bilinear(64, 64, 1)

        # Apply the weight initialization
        self.initialize_weights()

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def initialize_weights(self):
        for m in [self.bilinear_pool1, self.bilinear_pool2, self.bilinear_pool3]:
            self.weights_init(m)





    def forward(self, root_nodes, positive_subgraphs, malicious_subgraphs1):
        # --- 对正样本子图进行图卷积 ---
        positive_root_embeddings = []
        positive_pooled_embeddings = []  # 存储池化后的非根节点嵌入
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
            # x_feat = self.features[positive_nodes]
            # x_combined = torch.cat((x_embed, x_feat), dim=1)

            # 将拼接后的结果传入 MLP
            x_embed = self.feature_embedding_mlp(x_embed)

            # 图卷积操作
            x = self.gcn_conv1(x_embed, edge_index)
            # x = F.relu(x)
            # x = self.gcn_conv2(x, edge_index)


            x1 = self.gcn_conv2(x, edge_index)


            # 存储卷积后的节点嵌入
            # positive_node_embeddings_list.append(x)
            # positive_node_mappings_list.append(node_mapping)

            # 获取正样本根节点的嵌入
            positive_root_embedding = x1[node_mapping[root_nodes[idx]]]
            positive_root_embeddings.append(positive_root_embedding)

            # 计算非根节点的池化嵌入
            non_root_indices = [i for i in range(len(positive_nodes)) if i != node_mapping[root_nodes[idx]]]
            if non_root_indices:
                pooled_embedding = torch.mean(x[non_root_indices], dim=0)  # 平均池化
                positive_pooled_embeddings.append(pooled_embedding)

        # 将批量嵌入转化为张量
        positive_root_embeddings = torch.stack(positive_root_embeddings)  # [batch_size, hidden_dim]
        positive_pooled_embeddings = torch.stack(positive_pooled_embeddings)  # [batch_size, hidden_dim]

        # --- 使用 process_subgraphs_with_positive_embedding 处理恶意子图 ---
        # 调用函数时，不再使用 process_subgraphs_with_positive_embedding 返回的正样本嵌入
        # --- 处理恶意节点 ---
        # malicious_subgraphs1 不再是子图，而是单个恶意节点列表
        averaged_embeddings = []  # 存储根节点和恶意节点平均后的嵌入

        for idx, malicious_node in enumerate(malicious_subgraphs1):
            # 获取正样本卷积后的根节点嵌入
            root_embedding = positive_root_embeddings[idx]

            # 从 self.embeddings 提取恶意节点的嵌入
            malicious_embedding = self.embeddings[malicious_node]


            malicious_embedding = self.feature_embedding_mlp(malicious_embedding)

            # 计算根节点与恶意节点的平均嵌入
            avg_embedding = torch.mean(torch.stack([root_embedding, malicious_embedding]), dim=0)
            averaged_embeddings.append(avg_embedding)

        avg_embeddings = torch.stack(averaged_embeddings)

        # --- 假设您的模型在 GPU 上运行，获取 device ---
        device = self.embeddings.device  # 或者使用指定的 device，例如 torch.device("cuda")

        # --- 计算无噪声池化子图和虚拟节点的分数 ---

        # 扩展 virtual_node 并确保它在正确的设备上
        virtual_node_expanded1 = self.virtual_node1.expand(positive_pooled_embeddings.size(0), -1).to(device)
        # 要经过linear和act
        virtual_node_expanded1 = self.gcn_conv1(virtual_node_expanded1,edge_index)
        # 确保 positive_pooled_embeddings 在同一设备上
        positive_pooled_embeddings = positive_pooled_embeddings.to(device)

        pool_score = self.bilinear_pool1(virtual_node_expanded1, positive_pooled_embeddings)  # [batch_size, 1]

        # --- 对池化子图加上噪声 ---
        noise = self.single_noise1.expand(positive_pooled_embeddings.size(0), -1).to(device)  # [batch_size, 64]
        processed_noise = self.mlp_noise(noise)  # 处理后的噪声向量 [batch_size, hidden_dim]
        combined_with_noise = positive_pooled_embeddings + processed_noise  # 加上噪声后的池化子图嵌入

        # --- 计算对池化子图加噪声后和虚拟节点的分数 ---
        noise_pool_score = self.bilinear_pool1(virtual_node_expanded1, combined_with_noise)  # [batch_size, 1]


        # 扩展 virtual_node 并确保它在正确的设备上
        virtual_node_expanded = self.virtual_node.expand(positive_pooled_embeddings.size(0), -1).to(device)
        # 要经过linear和act
        virtual_node_expanded = self.gcn_conv2(virtual_node_expanded, edge_index)
        # --- 计算根节点和虚拟节点的分数 ---
        positive_root_embeddings = positive_root_embeddings.to(device)  # 将根节点嵌入转移到同一设备
        root_score = self.bilinear_pool2(virtual_node_expanded, positive_root_embeddings)  # [batch_size, 1]

        # --- 处理根节点嵌入并生成根节点的噪声 ---
        root_noise = self.single_noise.expand(positive_root_embeddings.size(0), -1).to(device)  # [batch_size, 64]
        processed_root_noise = self.mlp_noise1(root_noise)  # 处理后的根节点噪声
        combined_root_with_noise = positive_root_embeddings + processed_root_noise  # 加噪后的根节点嵌入

        # --- 计算加噪根节点和虚拟节点的分数 ---
        noise_root_score = self.bilinear_pool2(virtual_node_expanded, combined_root_with_noise)  # [batch_size, 1]

        # --- 计算 avg_embedding 和虚拟节点的分数（malicious_score）---
        avg_embeddings = avg_embeddings.to(device)  # 确保 avg_embeddings 在正确的设备上
        malicious_score = self.bilinear_pool3(virtual_node_expanded, avg_embeddings)  # 计算恶意节点平均嵌入和虚拟节点的分数

        return pool_score, noise_pool_score, root_score, noise_root_score, malicious_score, positive_pooled_embeddings

# # 初始化ADAG模型时，删除对 embeddings1 的操作
# adag_model = ADAG(input_dim=embeddings.shape[1], hidden_dim=128, output_dim=output_dim).to(device)
#
# # 将 DeepWalk 的嵌入赋值到唯一的 embeddings 变量中
# adag_model.embeddings = nn.Parameter(torch.FloatTensor(embeddings), requires_grad=True).to(device)
#
# # 继续使用相同的 optimizer
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, adag_model.parameters()), lr=0.001)
