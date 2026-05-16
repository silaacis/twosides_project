import json
import os
import torch
import gradio as gr
import pandas as pd
import pubchempy as pcp
import requests

from tdc.multi_pred import DDI
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

from graph_utils import smiles_to_graph
from model import SiameseGATv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESCRIPTION_FILE = "side_effect_descriptions_enhanced.json"


def load_side_effect_descriptions():
    if os.path.exists(DESCRIPTION_FILE):
        with open(DESCRIPTION_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    print("Uyarı: side_effect_descriptions_enhanced.json bulunamadı.")
    return {}


side_effect_descriptions = load_side_effect_descriptions()


def get_side_effect_info(label_id):
    key = str(label_id)

    if key in side_effect_descriptions:
        item = side_effect_descriptions[key]
        return {
            "en_name": item.get("en_name", key),
            "tr_name": item.get("tr_name", item.get("en_name", key)),
            "description": item.get("description", "Açıklama bulunamadı."),
            "source": item.get("source", "Açıklama dosyası"),
        }

    return {
        "en_name": key,
        "tr_name": key,
        "description": "Bu yan etki için açıklama bulunamadı.",
        "source": "Varsayılan",
    }


def get_pubchem_record_title(cid):
    try:
        pure_cid = str(int(str(cid).replace("CID", "")))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{pure_cid}/JSON"

        response = requests.get(url, timeout=8)

        if response.status_code == 200:
            data = response.json()
            title = data.get("Record", {}).get("RecordTitle", None)

            if title and not title.upper().startswith("CID"):
                return title.title()

    except Exception:
        pass

    return None


def get_drug_name(cid):
    try:
        pure_cid = int(str(cid).replace("CID", ""))

        title_name = get_pubchem_record_title(cid)
        if title_name:
            return title_name

        compound = pcp.Compound.from_cid(pure_cid)

        if compound.synonyms:
            clean_names = [
                name
                for name in compound.synonyms
                if len(name) < 50
                and not name.upper().startswith("CID")
                and not any(char.isdigit() for char in name[:5])
            ]

            if clean_names:
                return clean_names[0].title()

            return compound.synonyms[0].title()

        if compound.iupac_name:
            return compound.iupac_name.title()

    except Exception:
        pass

    return f"Bilinmeyen İlaç [{cid}]"


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
    weights_only=False,
)

num_classes = len(mlb.classes_)

print("Gerçek etiket sözlüğü hazırlanıyor...")

true_label_dict = {}

for _, row in grouped_df.iterrows():
    d1 = row["Drug1_ID"]
    d2 = row["Drug2_ID"]
    labels = set(str(label) for label in row["Y"])

    true_label_dict[(d1, d2)] = labels
    true_label_dict[(d2, d1)] = labels


print("İlaç isimleri hazırlanıyor...")

drug_dict = {}
display_to_id = {}
all_drugs = {}

for _, row in grouped_df.iterrows():
    all_drugs[row["Drug1_ID"]] = row["Drug1"]
    all_drugs[row["Drug2_ID"]] = row["Drug2"]

for cid, smiles in all_drugs.items():
    drug_name = get_drug_name(cid)

    if drug_name.startswith("Bilinmeyen İlaç"):
        display_name = drug_name
    else:
        display_name = f"{drug_name} [{cid}]"

    drug_dict[cid] = smiles
    display_to_id[display_name] = cid

drug_list = sorted(list(display_to_id.keys()))

model = SiameseGATv2(
    num_node_features=80,
    num_edge_features=6,
    hidden_dim=128,
    num_classes=num_classes,
).to(device)

model.load_state_dict(
    torch.load(
        "models/twosides_gatv2_model.pth",
        map_location=device,
    )
)

model.eval()

print("Model başarıyla yüklendi.")


def predict_side_effects(drug1_display, drug2_display):
    if drug1_display is None or drug2_display is None:
        empty_df = pd.DataFrame()
        return None, "Lütfen iki ilaç seçiniz.", empty_df, empty_df

    drug1_id = display_to_id[drug1_display]
    drug2_id = display_to_id[drug2_display]

    if drug1_id == drug2_id:
        empty_df = pd.DataFrame()
        return None, "Lütfen iki farklı ilaç seçiniz.", empty_df, empty_df

    smiles1 = drug_dict[drug1_id]
    smiles2 = drug_dict[drug2_id]

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        empty_df = pd.DataFrame()
        return None, "Molekül yapısı okunamadı.", empty_df, empty_df

    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        empty_df = pd.DataFrame()
        return None, "Graf dönüşümü başarısız oldu.", empty_df, empty_df

    batch1 = Batch.from_data_list([g1]).to(device)
    batch2 = Batch.from_data_list([g2]).to(device)

    with torch.no_grad():
        output = model(batch1, batch2)
        scores = torch.sigmoid(output)[0]

    top_scores, top_indices = torch.topk(scores, 10)

    true_labels = true_label_dict.get((drug1_id, drug2_id), set())

    image = Draw.MolsToGridImage(
        [mol1, mol2],
        legends=[drug1_display.split(" [")[0], drug2_display.split(" [")[0]],
        molsPerRow=2,
        subImgSize=(350, 300),
    )

    prediction_rows = []
    matched_count = 0

    for rank, idx in enumerate(top_indices, start=1):
        label_id = str(mlb.classes_[idx.item()])
        score = top_scores[rank - 1].item() * 100

        info = get_side_effect_info(label_id)
        is_match = label_id in true_labels

        if is_match:
            matched_count += 1

        prediction_rows.append(
            {
                "Sıra": rank,
                "Yan Etki": info["en_name"],
                "Türkçe Karşılık": info["tr_name"],
                "Açıklama": info["description"],
                "Kaynak": info["source"],
                "Model Skoru": f"%{score:.2f}",
                "Veri Setinde Var mı?": "Evet" if is_match else "Hayır",
            }
        )

    prediction_df = pd.DataFrame(prediction_rows)

    true_rows = []

    for label_id in sorted(true_labels):
        info = get_side_effect_info(label_id)
        true_rows.append(
            {
                "Yan Etki": info["en_name"],
                "Türkçe Karşılık": info["tr_name"],
                "Açıklama": info["description"],
            }
        )

    true_df = pd.DataFrame(true_rows)

    precision_at_10 = matched_count / 10

    summary = (
        f"### Değerlendirme Özeti\n\n"
        f"- **Top-10 eşleşme:** {matched_count} / 10\n"
        f"- **Precision@10:** {precision_at_10:.2f}\n"
        f"- **Veri setindeki gerçek yan etki sayısı:** {len(true_labels)}\n\n"
        f"Model skorları kesin klinik olasılık değil, TWOSIDES veri setinden öğrenilen örüntülere dayalı göreli tahmin skorlarıdır."
    )

    return image, summary, prediction_df, true_df


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi")

    gr.Markdown(
        "Bu arayüz, iki ilacın birlikte kullanımında ortaya çıkabilecek olası yan etkileri "
        "TWOSIDES veri seti üzerinde eğitilmiş GATv2 tabanlı GNN modeliyle tahmin eder."
    )

    with gr.Row():
        with gr.Column(scale=1):
            drug1 = gr.Dropdown(
                choices=drug_list,
                label="Birinci İlaç",
                filterable=True,
            )

            drug2 = gr.Dropdown(
                choices=drug_list,
                label="İkinci İlaç",
                filterable=True,
            )

            button = gr.Button("Yan Etki Tahmini Yap", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Moleküler Yapılar")

    output_summary = gr.Markdown()

    gr.Markdown("## Modelin Tahmin Ettiği İlk 10 Yan Etki")

    prediction_table = gr.Dataframe(
        headers=[
            "Sıra",
            "Yan Etki",
            "Türkçe Karşılık",
            "Açıklama",
            "Kaynak",
            "Model Skoru",
            "Veri Setinde Var mı?",
        ],
        interactive=False,
        wrap=True,
    )

    gr.Markdown("## TWOSIDES Veri Setindeki Gerçek Yan Etkiler")

    true_table = gr.Dataframe(
        headers=[
            "Yan Etki",
            "Türkçe Karşılık",
            "Açıklama",
        ],
        interactive=False,
        wrap=True,
    )

    gr.Markdown(
        "**Not:** Bu sistem tıbbi tanı veya tedavi önerisi değildir. "
        "Sonuçlar araştırma/prototip amaçlıdır."
    )

    button.click(
        predict_side_effects,
        inputs=[drug1, drug2],
        outputs=[
            output_image,
            output_summary,
            prediction_table,
            true_table,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)