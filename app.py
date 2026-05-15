import torch
import gradio as gr
import pandas as pd
from tdc.multi_pred import DDI
from tdc.utils import get_label_map
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

from graph_utils import smiles_to_graph
from model import SiameseGATv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Kullanılan cihaz: {device}")
print("TWOSIDES verisi yükleniyor...")

data = DDI(name="TWOSIDES")
df = data.get_data()

grouped_df = (
    df.groupby(["Drug1_ID", "Drug1", "Drug2_ID", "Drug2"])["Y"]
    .apply(list)
    .reset_index()
)

mlb = torch.load(
    "models/label_binarizer.pth",
    map_location=device,
    weights_only=False
)
num_classes = len(mlb.classes_)

label_map = get_label_map(
    name="TWOSIDES",
    task="DDI",
    name_column="Side Effect Name"
)

drug_dict = {}

for _, row in grouped_df.iterrows():
    drug_dict[row["Drug1_ID"]] = row["Drug1"]
    drug_dict[row["Drug2_ID"]] = row["Drug2"]

drug_list = sorted(list(drug_dict.keys()))

model = SiameseGATv2(
    num_node_features=80,
    num_edge_features=6,
    hidden_dim=128,
    num_classes=num_classes,
).to(device)

model.load_state_dict(
    torch.load("models/twosides_gatv2_model.pth", map_location=device)
)

model.eval()

print("Model başarıyla yüklendi.")


def predict_side_effects(drug1_id, drug2_id):
    if drug1_id == drug2_id:
        return None, "Lütfen iki farklı ilaç seçiniz."

    smiles1 = drug_dict[drug1_id]
    smiles2 = drug_dict[drug2_id]

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None, "Molekül yapısı okunamadı."

    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return None, "Graf dönüşümü başarısız oldu."

    batch1 = Batch.from_data_list([g1]).to(device)
    batch2 = Batch.from_data_list([g2]).to(device)

    with torch.no_grad():
        output = model(batch1, batch2)
        probs = torch.sigmoid(output)[0]

    top_probs, top_indices = torch.topk(probs, 10)

    image = Draw.MolsToGridImage(
        [mol1, mol2],
        legends=[drug1_id, drug2_id],
        molsPerRow=2,
        subImgSize=(350, 300),
    )

    result = "## Tahmin Edilen İlk 10 Yan Etki\n\n"
    result += "| Sıra | Yan Etki | Olasılık |\n"
    result += "|---|---|---|\n"

    for i, idx in enumerate(top_indices):
        label_id = mlb.classes_[idx.item()]
        side_effect_name = label_map.get(label_id, str(label_id))
        probability = top_probs[i].item() * 100

        result += f"| {i+1} | {side_effect_name} | %{probability:.2f} |\n"

    result += "\n---\n"
    result += "**Not:** Bu sistem tıbbi tanı veya tedavi önerisi değildir. "
    result += "Model yalnızca TWOSIDES veri setindeki örüntülere dayalı tahmin üretir."

    return image, result


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi")
    gr.Markdown(
        "Bu arayüz, iki ilacın birlikte kullanımında ortaya çıkabilecek "
        "olası yan etkileri TWOSIDES veri seti üzerinde eğitilmiş GATv2 modeliyle tahmin eder."
    )

    with gr.Row():
        with gr.Column():
            drug1 = gr.Dropdown(
                choices=drug_list,
                label="Birinci İlaç ID",
                filterable=True,
            )
            drug2 = gr.Dropdown(
                choices=drug_list,
                label="İkinci İlaç ID",
                filterable=True,
            )
            button = gr.Button("Yan Etki Tahmini Yap")

        with gr.Column():
            output_image = gr.Image(label="Moleküler Yapılar")
            output_text = gr.Markdown()

    button.click(
        predict_side_effects,
        inputs=[drug1, drug2],
        outputs=[output_image, output_text],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)