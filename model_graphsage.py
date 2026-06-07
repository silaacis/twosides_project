"""
Bu dosya, Siamese GraphSAGE modelini içermektedir.

Modelin amacı:

1. Birinci ilacın moleküler graphını analiz etmek
2. İkinci ilacın moleküler graphını analiz etmek
3. Her ilaç için bir embedding (sayısal temsil) üretmek
4. İki embedding'i birleştirmek
5. 1317 olası yan etki için tahmin skoru üretmek

Model Yapısı:

- GraphSAGE Katmanı 1
- Batch Normalization
- ReLU Aktivasyon
- Dropout

- GraphSAGE Katmanı 2
- Batch Normalization
- ReLU Aktivasyon

- Global Mean Pooling
- Global Max Pooling

- Tam Bağlantılı (Fully Connected) Sınıflandırıcı

Siamese Mimari:

Aynı GraphSAGE ağı hem birinci hem ikinci ilacı işler.
Böylece iki molekül aynı özellik uzayında temsil edilir.

Çıktı:

Model her ilaç çifti için 1317 yan etki sınıfına ait
olasılık skorları üretmektedir.

Final çalışmada yapılan model karşılaştırmaları sonucunda
GraphSAGE modeli GCN ve GATv2 modellerinden daha yüksek
ROC-AUC ve PR-AUC performansı göstermiştir ve final model
olarak seçilmiştir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool


class SiameseGraphSAGE(nn.Module):
    def __init__(
        self,
        num_node_features,
        num_edge_features,
        hidden_dim,
        num_classes,
    ):
        super(SiameseGraphSAGE, self).__init__()

        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
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

    def forward(self, drug1_data, drug2_data):
        emb1 = self.get_embedding(drug1_data)
        emb2 = self.get_embedding(drug2_data)

        combined = torch.cat([emb1, emb2], dim=1)

        return self.classifier(combined)