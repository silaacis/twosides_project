import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from tdc.multi_pred import DDI
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from graph_utils import smiles_to_graph
from model import SiameseGATv2


warnings.filterwarnings("ignore")


class TwosidesDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        g1 = smiles_to_graph(row["Drug1"])
        g2 = smiles_to_graph(row["Drug2"])
        label = row["Y_tensor"]

        return g1, g2, label


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        g1, g2, labels = batch

        g1 = g1.to(device)
        g2 = g2.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(g1, g2)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 200 == 0:
            print(
                f"[Batch {batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / len(train_loader)


def evaluate(model, data_loader, device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in data_loader:
            g1, g2, labels = batch

            g1 = g1.to(device)
            g2 = g2.to(device)
            labels = labels.to(device)

            outputs = model(g1, g2)
            probs = torch.sigmoid(outputs)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    try:
        roc_auc = roc_auc_score(all_labels, all_preds, average="micro")
        pr_auc = average_precision_score(all_labels, all_preds, average="micro")
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0

    return roc_auc, pr_auc


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    print("TWOSIDES veri seti indiriliyor/yükleniyor...")
    data = DDI(name="TWOSIDES")
    df = data.get_data()

    print("İlaç çiftleri gruplanıyor...")
    grouped_df = (
        df.groupby(["Drug1_ID", "Drug1", "Drug2_ID", "Drug2"])["Y"]
        .apply(list)
        .reset_index()
    )

    print(f"Toplam benzersiz ilaç çifti: {len(grouped_df)}")

    mlb = MultiLabelBinarizer()
    y_multilabel = mlb.fit_transform(grouped_df["Y"])

    grouped_df["Y_tensor"] = [
        torch.tensor(y, dtype=torch.float32) for y in y_multilabel
    ]

    num_classes = len(mlb.classes_)
    print(f"Toplam yan etki sınıfı: {num_classes}")

    train_val_df, test_df = train_test_split(
        grouped_df,
        test_size=0.2,
        random_state=42,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,
        random_state=42,
    )

    print(f"Train örnek sayısı: {len(train_df)}")
    print(f"Validation örnek sayısı: {len(val_df)}")
    print(f"Test örnek sayısı: {len(test_df)}")

    batch_size = 64

    train_loader = DataLoader(
        TwosidesDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        TwosidesDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        TwosidesDataset(test_df),
        batch_size=batch_size,
        shuffle=False,
    )

    num_node_features = 80
    num_edge_features = 6
    hidden_dim = 128

    model = SiameseGATv2(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    epochs = 10
    best_val_pr_auc = 0.0

    history = []

    torch.save(mlb, "models/label_binarizer.pth")

    print("Eğitim başlıyor...")
    print("-" * 95)
    print(
        f"{'Epoch':<8} "
        f"{'Train Loss':<15} "
        f"{'Val ROC-AUC':<15} "
        f"{'Val PR-AUC':<15} "
        f"{'Best PR-AUC':<15} "
        f"{'Süre(sn)':<10}"
    )
    print("-" * 95)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_roc_auc, val_pr_auc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        scheduler.step(val_pr_auc)

        duration = time.time() - start_time

        is_best = val_pr_auc > best_val_pr_auc

        if is_best:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), "models/twosides_gatv2_best_model.pth")
            torch.save(model.state_dict(), "models/twosides_gatv2_model.pth")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_val_pr_auc": best_val_pr_auc,
                "duration_seconds": duration,
                "is_best": is_best,
            }
        )

        pd.DataFrame(history).to_csv(
            "results/training_history.csv",
            index=False,
            encoding="utf-8",
        )

        print(
            f"{epoch:<8} "
            f"{train_loss:<15.4f} "
            f"{val_roc_auc:<15.4f} "
            f"{val_pr_auc:<15.4f} "
            f"{best_val_pr_auc:<15.4f} "
            f"{duration:<10.0f}"
        )

    print("-" * 95)
    print("En iyi model validation PR-AUC değerine göre kaydedildi.")
    print("Final test değerlendirmesi yapılıyor...")

    model.load_state_dict(
        torch.load(
            "models/twosides_gatv2_best_model.pth",
            map_location=device,
        )
    )

    test_roc_auc, test_pr_auc = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    final_results = {
        "best_val_pr_auc": best_val_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "num_classes": num_classes,
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "test_size": len(test_df),
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
    }

    pd.DataFrame([final_results]).to_csv(
        "results/final_test_results.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"Test PR-AUC: {test_pr_auc:.4f}")

    print("Eğitim tamamlandı.")
    print("Kaydedilen dosyalar:")
    print("- models/twosides_gatv2_model.pth")
    print("- models/twosides_gatv2_best_model.pth")
    print("- models/label_binarizer.pth")
    print("- results/training_history.csv")
    print("- results/final_test_results.csv")


if __name__ == "__main__":
    main()