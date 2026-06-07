"""
model_graphsage_fp.py

Bu dosya, GraphSAGE modelinin Morgan Fingerprint özellikleri ile
zenginleştirilmiş sürümünü içerir.

Temel fikir:

- GraphSAGE molekülün graph yapısından embedding üretir.
- Morgan Fingerprint molekülün kimyasal alt yapılarını özetler.
- Bu iki temsil birleştirilerek daha zengin bir ilaç temsili elde edilir.

Model akışı:

İlaç 1:
GraphSAGE embedding + Morgan Fingerprint

İlaç 2:
GraphSAGE embedding + Morgan Fingerprint

Sonra iki ilaç temsili birleştirilir ve 1317 yan etki için skor üretilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool


class SiameseGraphSAGEFingerprint(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_dim,
        num_classes,
        fingerprint_dim=512,
    ):
        super(SiameseGraphSAGEFingerprint, self).__init__()

        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.3)

        combined_drug_dim = hidden_dim + fingerprint_dim

        self.classifier = nn.Sequential(
            nn.Linear(combined_drug_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            self.dropout,

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            self.dropout,

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes),
        )

    def get_graph_embedding(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        return x_mean + x_max

    def forward(self, drug1_data, drug2_data, fp1, fp2):
        emb1 = self.get_graph_embedding(drug1_data)
        emb2 = self.get_graph_embedding(drug2_data)

        drug1_representation = torch.cat([emb1, fp1], dim=1)
        drug2_representation = torch.cat([emb2, fp2], dim=1)

        combined = torch.cat(
            [drug1_representation, drug2_representation],
            dim=1,
        )

        return self.classifier(combined)