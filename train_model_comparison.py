import os
import time
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from tdc.multi_pred import DDI

from graph_utils import smiles_to_graph

from model import SiameseGATv2
from model_gcn import SiameseGCN
from model_graphsage import SiameseGraphSAGE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Kullanılan cihaz: {device}")


class DrugPairDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.dataframe = dataframe.reset_index(drop=True)
        self.labels = labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        graph1 = smiles_to_graph(row["Drug1"])
        graph2 = smiles_to_graph(row["Drug2"])

        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return graph1, graph2, label


def collate_fn(batch):
    graphs1, graphs2, labels = zip(*batch)

    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)

    labels = torch.stack(labels)

    return batch1, batch2, labels


print("TWOSIDES veri seti yükleniyor...")

data = DDI(name="TWOSIDES")
df = data.get_data()

grouped_df = (
    df.groupby(["Drug1_ID", "Drug1", "Drug2_ID", "Drug2"])["Y"]
    .apply(list)
    .reset_index()
)

print(f"Toplam benzersiz ilaç çifti: {len(grouped_df)}")

mlb = MultiLabelBinarizer()

y_multilabel = mlb.fit_transform(grouped_df["Y"])

num_classes = len(mlb.classes_)

print(f"Toplam yan etki sınıfı: {num_classes}")

train_df, temp_df, train_labels, temp_labels = train_test_split(
    grouped_df,
    y_multilabel,
    test_size=0.30,
    random_state=42,
)

val_df, test_df, val_labels, test_labels = train_test_split(
    temp_df,
    temp_labels,
    test_size=0.67,
    random_state=42,
)

print(f"Train örnek sayısı: {len(train_df)}")
print(f"Validation örnek sayısı: {len(val_df)}")
print(f"Test örnek sayısı: {len(test_df)}")

train_dataset = DrugPairDataset(train_df, train_labels)
val_dataset = DrugPairDataset(val_df, val_labels)
test_dataset = DrugPairDataset(test_df, test_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn,
)


def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch1, batch2, labels in loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            labels = labels.to(device)

            outputs = model(batch1, batch2)

            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    roc_auc = roc_auc_score(y_true, y_pred, average="macro")
    pr_auc = average_precision_score(y_true, y_pred, average="macro")

    return roc_auc, pr_auc


def train_single_model(model_name, model_class):
    print("\n" + "=" * 70)
    print(f"{model_name} eğitimi başlıyor...")
    print("=" * 70)

    model = model_class(
        num_node_features=80,
        num_edge_features=6,
        hidden_dim=128,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.BCEWithLogitsLoss()

    best_pr_auc = 0

    for epoch in range(1, 3):
        model.train()

        total_loss = 0

        start_time = time.time()

        progress_bar = tqdm(train_loader)

        for batch1, batch2, labels in progress_bar:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch1, batch2)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch} | Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        val_roc_auc, val_pr_auc = evaluate_model(model, val_loader)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Süre: {elapsed:.1f} sn"
        )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc

            torch.save(
                model.state_dict(),
                f"models/{model_name.lower()}_best.pth",
            )

    print(f"\n{model_name} test değerlendirmesi yapılıyor...")

    model.load_state_dict(
        torch.load(
            f"models/{model_name.lower()}_best.pth",
            map_location=device,
        )
    )

    test_roc_auc, test_pr_auc = evaluate_model(model, test_loader)

    print(f"{model_name} Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"{model_name} Test PR-AUC: {test_pr_auc:.4f}")

    return {
        "Model": model_name,
        "Test ROC-AUC": round(test_roc_auc, 4),
        "Test PR-AUC": round(test_pr_auc, 4),
    }


results = []

results.append(
    train_single_model("GCN", SiameseGCN)
)

results.append(
    train_single_model("GraphSAGE", SiameseGraphSAGE)
)

results.append(
    train_single_model("GATv2", SiameseGATv2)
)

results_df = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False,
)

print("\nKarşılaştırma sonuçları:")
print(results_df)

print("\nSonuçlar kaydedildi:")
print("results/model_comparison.csv")