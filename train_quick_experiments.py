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
    def __init__(self, dataframe, labels, fingerprint_dim=512):
        self.dataframe = dataframe.reset_index(drop=True)
        self.labels = labels
        self.fingerprint_dim = fingerprint_dim

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        smiles1 = row["Drug1"]
        smiles2 = row["Drug2"]

        graph1 = smiles_to_graph(smiles1)
        graph2 = smiles_to_graph(smiles2)

        fp1 = smiles_to_morgan_fingerprint(smiles1, n_bits=self.fingerprint_dim)
        fp2 = smiles_to_morgan_fingerprint(smiles2, n_bits=self.fingerprint_dim)

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

    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    pr_auc = average_precision_score(y_true, y_pred, average="micro")

    return roc_auc, pr_auc


def calculate_pos_weight(labels):
    positive_counts = labels.sum(axis=0)
    negative_counts = labels.shape[0] - positive_counts

    pos_weight = negative_counts / (positive_counts + 1e-6)
    pos_weight = np.clip(pos_weight, 1.0, 20.0)

    return torch.tensor(pos_weight, dtype=torch.float).to(device)


def run_experiment(
    experiment_name,
    train_df,
    val_df,
    test_df,
    train_labels,
    val_labels,
    test_labels,
    num_classes,
    fingerprint_dim=512,
    epochs=5,
    use_weighted_bce=False,
):
    print("\n" + "=" * 80)
    print(f"Deney başlıyor: {experiment_name}")
    print("=" * 80)

    batch_size = 64

    train_loader = DataLoader(
        DrugPairFingerprintDataset(train_df, train_labels, fingerprint_dim),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        DrugPairFingerprintDataset(val_df, val_labels, fingerprint_dim),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        DrugPairFingerprintDataset(test_df, test_labels, fingerprint_dim),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = SiameseGraphSAGEFingerprint(
        num_node_features=80,
        num_edge_features=6,
        hidden_dim=128,
        num_classes=num_classes,
        fingerprint_dim=fingerprint_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    if use_weighted_bce:
        pos_weight = calculate_pos_weight(train_labels)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    best_val_pr_auc = 0.0
    best_model_path = f"models/{experiment_name}_best.pth"

    history = []

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
                f"{experiment_name} | Epoch {epoch} | Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        val_roc_auc, val_pr_auc = evaluate_model(model, val_loader)

        duration = time.time() - start_time

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), best_model_path)

        history.append(
            {
                "experiment": experiment_name,
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_val_pr_auc": best_val_pr_auc,
                "duration_seconds": duration,
            }
        )

        print(
            f"Epoch {epoch} | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | "
            f"Val PR-AUC: {val_pr_auc:.4f} | "
            f"Best PR-AUC: {best_val_pr_auc:.4f} | "
            f"Süre: {duration:.1f} sn"
        )

    pd.DataFrame(history).to_csv(
        f"results/{experiment_name}_history.csv",
        index=False,
        encoding="utf-8",
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
        )
    )

    test_roc_auc, test_pr_auc = evaluate_model(model, test_loader)

    print(f"{experiment_name} Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"{experiment_name} Test PR-AUC: {test_pr_auc:.4f}")

    return {
        "experiment": experiment_name,
        "fingerprint_dim": fingerprint_dim,
        "epochs": epochs,
        "loss": "Weighted BCE" if use_weighted_bce else "BCE",
        "best_val_pr_auc": round(best_val_pr_auc, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
    }


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

    experiments = [
        {
            "experiment_name": "fp1024_bce_5epoch",
            "fingerprint_dim": 1024,
            "epochs": 5,
            "use_weighted_bce": False,
        },
        {
            "experiment_name": "fp512_weighted_bce_5epoch",
            "fingerprint_dim": 512,
            "epochs": 5,
            "use_weighted_bce": True,
        },
        {
            "experiment_name": "fp512_bce_5epoch",
            "fingerprint_dim": 512,
            "epochs": 5,
            "use_weighted_bce": False,
        },
    ]

    results = []

    for exp in experiments:
        result = run_experiment(
            experiment_name=exp["experiment_name"],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
            num_classes=num_classes,
            fingerprint_dim=exp["fingerprint_dim"],
            epochs=exp["epochs"],
            use_weighted_bce=exp["use_weighted_bce"],
        )

        results.append(result)

        pd.DataFrame(results).to_csv(
            "results/quick_experiment_results.csv",
            index=False,
            encoding="utf-8",
        )

    results_df = pd.DataFrame(results)

    print("\nHızlı deney sonuçları:")
    print(results_df)

    print("\nSonuç dosyası:")
    print("- results/quick_experiment_results.csv")


if __name__ == "__main__":
    main()