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
from model_gat import SiameseGATv2
from model_gcn import SiameseGCN
from model_graphsage import SiameseGraphSAGE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


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


def train_single_model(
    model_name,
    model_class,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    epochs=10,
):
    print("\n" + "=" * 70)
    print(f"{model_name} eğitimi başlıyor...")
    print("=" * 70)

    model = model_class(
        num_node_features=80,
        num_edge_features=6,
        hidden_dim=128,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_pr_auc = 0.0
    best_model_path = f"models/{model_name.lower()}_best.pth"

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
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
                f"{model_name} | Epoch {epoch} | Loss: {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        val_roc_auc, val_pr_auc = evaluate_model(model, val_loader)

        duration = time.time() - start_time

        is_best = val_pr_auc > best_val_pr_auc

        if is_best:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), best_model_path)

        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
                "best_val_pr_auc": best_val_pr_auc,
                "duration_seconds": duration,
                "is_best": is_best,
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
        f"results/{model_name.lower()}_comparison_history.csv",
        index=False,
        encoding="utf-8",
    )

    print(f"\n{model_name} test değerlendirmesi yapılıyor...")

    if os.path.exists(best_model_path):
        model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=device,
            )
        )

    test_roc_auc, test_pr_auc = evaluate_model(model, test_loader)

    print(f"{model_name} Test ROC-AUC: {test_roc_auc:.4f}")
    print(f"{model_name} Test PR-AUC: {test_pr_auc:.4f}")

    return {
        "Model": model_name,
        "Epoch": epochs,
        "Best Val PR-AUC": round(best_val_pr_auc, 4),
        "Test ROC-AUC": round(test_roc_auc, 4),
        "Test PR-AUC": round(test_pr_auc, 4),
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

    batch_size = 64

    train_loader = DataLoader(
        DrugPairDataset(train_df, train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        DrugPairDataset(val_df, val_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        DrugPairDataset(test_df, test_labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    comparison_results = []

    comparison_results.append(
        train_single_model(
            model_name="GCN",
            model_class=SiameseGCN,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            epochs=10,
        )
    )

    comparison_results.append(
        train_single_model(
            model_name="GraphSAGE",
            model_class=SiameseGraphSAGE,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            epochs=10,
        )
    )

    comparison_results.append(
        train_single_model(
            model_name="GATv2",
            model_class=SiameseGATv2,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
            epochs=10,
        )
    )

    results_df = pd.DataFrame(comparison_results)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nKarşılaştırma sonuçları:")
    print(results_df)

    print("\nSonuçlar kaydedildi:")
    print("- results/model_comparison.csv")
    print("- results/gcn_comparison_history.csv")
    print("- results/graphsage_comparison_history.csv")
    print("- results/gatv2_comparison_history.csv")


if __name__ == "__main__":
    main()