import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool


class SiameseGATv2(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_dim,
        num_classes,
        heads=4,
    ):
        super(SiameseGATv2, self).__init__()

        self.conv1 = GATv2Conv(
            num_node_features,
            hidden_dim,
            heads=heads,
            edge_dim=num_edge_features,
            concat=True,
        )

        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)

        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            edge_dim=num_edge_features,
            concat=False,
        )

        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def get_embedding(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return x_mean + x_max

    def forward(self, drug1_data, drug2_data):
        emb1 = self.get_embedding(drug1_data)
        emb2 = self.get_embedding(drug2_data)

        combined = torch.cat([emb1, emb2], dim=1)

        return self.classifier(combined)