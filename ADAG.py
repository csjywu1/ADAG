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


        self.virtual_node = None

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

        self.score_mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出一个分数
            nn.Sigmoid()  # 将分数归一化到 [0, 1] 范围内
        )

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def initialize_weights(self):
        for m in [self.bilinear_pool1, self.bilinear_pool2, self.bilinear_pool3]:
            self.weights_init(m)

    def add_noise(self, features, noise_level=0.1):
        # Add Gaussian noise to the features
        noise = torch.randn_like(features) * noise_level
        return features + noise

    def contrasting_loss(self, score_ab, score_ac, score_a_b, score_a_c, score_a_noisy, score_b_noisy):
        # Implement the contrasting loss calculation
        positive_loss = F.relu(1 - score_ab + score_ac).mean() + \
                        F.relu(1 - score_a_b + score_a_c).mean()
        negative_loss = F.relu(1 + score_a_noisy - score_b_noisy).mean()

        total_loss = positive_loss + negative_loss
        return total_loss

    def similarity_score(self, node1, node2):
        # Example similarity function (can be removed if using MLP only)
        return F.cosine_similarity(node1, node2)

    def forward(self, batch_features_a, batch_features_b, batch_features_c,
                batch_neighbors_a, batch_neighbors_b, batch_neighbors_c):
        # Get embeddings for all features
        self.embeddings = self.feature_embedding_mlp(self.features)

        # Compute the virtual node by pooling embeddings
        self.virtual_node = torch.mean(self.embeddings, dim=0, keepdim=True)

        # Calculate aggregated neighbor representations
        aggregated_neighbors_b = []
        aggregated_neighbors_c = []

        for i in range(batch_features_a.size(0)):
            # Retrieve indices for neighbors
            neighbors_b = batch_neighbors_b[i]
            neighbors_c = batch_neighbors_c[i]

            # Calculate similarity scores
            scores_b = self.similarity_score(self.embeddings[batch_features_a[i]], self.embeddings[neighbors_b])
            scores_c = self.similarity_score(self.embeddings[batch_features_a[i]], self.embeddings[neighbors_c])

            # Normalize scores to get weights
            weights_b = F.softmax(scores_b, dim=0)
            weights_c = F.softmax(scores_c, dim=0)

            # Weighted aggregation of neighbors
            aggregated_b = (weights_b.unsqueeze(1) * self.embeddings[neighbors_b]).sum(dim=0, keepdim=True)
            aggregated_c = (weights_c.unsqueeze(1) * self.embeddings[neighbors_c]).sum(dim=0, keepdim=True)

            aggregated_neighbors_b.append(aggregated_b)
            aggregated_neighbors_c.append(aggregated_c)

        # Stack aggregated results
        b1 = torch.cat(aggregated_neighbors_b, dim=0)
        c1 = torch.cat(aggregated_neighbors_c, dim=0)

        # Compute scores using MLP
        score_ab = self.score_mlp(torch.cat([self.virtual_node.expand_as(b1), b1], dim=1))
        score_ac = self.score_mlp(torch.cat([self.virtual_node.expand_as(c1), c1], dim=1))

        loss1 = score_ab - score_ac

        # MLP for (virtual node || (a, b)) and (virtual node || (a, c))
        score_a_b = self.score_mlp(torch.cat([self.virtual_node.expand_as(self.embeddings[batch_features_a]),
                                              self.embeddings[batch_features_a]], dim=1))
        score_a_c = self.score_mlp(torch.cat([self.virtual_node.expand_as(self.embeddings[batch_features_c]),
                                              self.embeddings[batch_features_c]], dim=1))
        loss2 = score_a_b + score_a_c

        # Adding noise
        score_a_noisy = self.score_mlp(torch.cat([self.virtual_node.expand_as(self.embeddings[batch_features_a]),
                                                  self.add_noise(self.embeddings[batch_features_a])], dim=1))
        score_b_noisy = self.score_mlp(torch.cat([self.virtual_node.expand_as(b1),
                                                  self.add_noise(b1)], dim=1))

        # Compute contrasting loss
        total_loss = self.contrasting_loss(score_ab, score_ac, score_a_b, score_a_c, score_a_noisy, score_b_noisy)

        return total_loss









