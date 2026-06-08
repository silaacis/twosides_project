"""
train_graphsage_fp.py

Bu dosya, GraphSAGE modelinin Morgan Fingerprint ile zenginleştirilmiş
versiyonunu eğitmek için kullanılır.

Amaç:
Standart GraphSAGE modeli sadece moleküler graph bilgisini kullanırken,
bu model hem graph embedding hem de Morgan Fingerprint bilgisini birlikte kullanır.

Model:
GraphSAGE + Morgan Fingerprint

Çıktılar:
- models/graphsage_fp_best.pth
- results/graphsage_fp_history.csv
- results/graphsage_fp_results.csv
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

from tdc.multi_pred import DDI

from graph_utils import smiles_to_graph
from fingerprint_utils import smiles_to_morgan_fingerprint
from model_graphsage_fp import SiameseGraphSAGEFingerprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


class DrugPairFingerprintDataset(Dataset):
    def __init__(self, dataframe, labels):
        self.dataframe = dataframe.reset_index(drop=True)
        self.labels = labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        smiles1 = row["Drug1"]
        smiles2 = row["Drug2"]

        graph1 = smiles_to_graph(smiles1)
        graph2 = smiles_to_graph(smiles2)

        fp1 = smiles_to_morgan_fingerprint(smiles1, n_bits=1024)
        fp2 = smiles_to_morgan_fingerprint(smiles2, n_bits=1024)

        label = torch.tensor(self.labels[idx], dtype=torch.float)

        return graph1, graph2, fp1, fp2, label


def collate_fn(batch):
    graphs1, graphs2, fps1, fps2, labels = zip(*batch)

    batch1 = Batch.from_data_list(graphs1)
    batch2 = Batch.from_data_list(graphs2)

    fps1 = torch.stack(fps1)
    fps2 = torch.stack(fps2)
    labels = torch.stack(labels)

    return batch1, batch2, fps1, fps2, labels


def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch1, batch2, fp1, fp2, labels in loader:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            labels = labels.to(device)

            outputs = model(batch1, batch2, fp1, fp2)
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    try:
        roc_auc = roc_auc_score(y_true, y_pred, average="micro")
        pr_auc = average_precision_score(y_true, y_pred, average="micro")
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    return roc_auc, pr_auc


def main():
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

    train_val_df, test_df, train_val_labels, test_labels = train_test_split(
        grouped_df,
        y_multilabel,
        test_size=0.2,
        random_state=42,
    )

    train_df, val_df, train_labels, val_labels = train_test_split(
        train_val_df,
        train_val_labels,
        test_size=0.125,
        random_state=42,
    )

    print(f"Train örnek sayısı: {len(train_df)}")
    print(f"Validation örnek sayısı: {len(val_df)}")
    print(f"Test örnek sayısı: {len(test_df)}")

    batch_size = 64

    train_loader = DataLoader(
        DrugPairFingerprintDataset(train_df, train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        DrugPairFingerprintDataset(val_df, val_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        DrugPairFingerprintDataset(test_df, test_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SiameseGraphSAGEFingerprint(
        num_node_features=80,
        num_edge_features=6,
        hidden_dim=128,
        num_classes=num_classes,
        fingerprint_dim=1024,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    epochs = 20
    best_val_pr_auc = 0.0

    history = []

    print("GraphSAGE + Morgan Fingerprint eğitimi başlıyor...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        progress_bar = tqdm(train_loader)

        for batch1, batch2, fp1, fp2, labels in progress_bar:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch1, batch2, fp1, fp2)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"GraphSAGE+FP | Epoch {epoch} | Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        val_roc_auc, val_pr_auc = evaluate_model(model, val_loader)

        duration = time.time() - start_time

        is_best = val_pr_auc > best_val_pr_auc

        if is_best:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), "models/graphsage_fp_best.pth")

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_val_pr_auc": best_val_pr_auc,
                "duration_seconds": duration,
                "is_best": is_best,
            }
        )

        pd.DataFrame(history).to_csv(
            "results/graphsage_fp_history.csv",
            index=False,
            encoding="utf-8",
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Best PR-AUC: {best_val_pr_auc:.4f} | "
            f"Süre: {duration:.1f} sn"
        )

    print("Final test değerlendirmesi yapılıyor...")

    model.load_state_dict(
        torch.load(
            "models/graphsage_fp_best.pth",
            map_location=device,
        )
    )

    test_roc_auc, test_pr_auc = evaluate_model(model, test_loader)

    final_results = {
        "model": "GraphSAGE + Morgan Fingerprint",
        "epochs": epochs,
        "best_val_pr_auc": best_val_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "num_classes": num_classes,
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "batch_size": batch_size,
        "hidden_dim": 128,
        "fingerprint_dim": 1024,
    }

    pd.DataFrame([final_results]).to_csv(
        "results/graphsage_fp_results.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"GraphSAGE + FP Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"GraphSAGE + FP Test PR-AUC: {test_pr_auc:.4f}")

    print("Kaydedilen dosyalar:")
    print("- models/graphsage_fp_best.pth")
    print("- results/graphsage_fp_history.csv")
    print("- results/graphsage_fp_results.csv")


if __name__ == "__main__":
    main()