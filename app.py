import json
import os
import torch
import gradio as gr
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

    print("Uyarı: side_effect_descriptions.json bulunamadı.")
    return {}


side_effect_descriptions = load_side_effect_descriptions()


def get_side_effect_info(label_id, effect_name):
    key = str(label_id)

    if key in side_effect_descriptions:
        item = side_effect_descriptions[key]
        return (
            item.get("tr_name", effect_name),
            item.get("description", "Açıklama bulunamadı."),
            item.get("source", "Açıklama dosyası"),
        )

    return (
        effect_name,
        f"{effect_name}, TWOSIDES veri setinde ilaç kombinasyonlarıyla ilişkili olarak bildirilen klinik bir yan etkidir.",
        "Varsayılan",
    )


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
        return None, "Lütfen iki ilaç seçiniz."

    drug1_id = display_to_id[drug1_display]
    drug2_id = display_to_id[drug2_display]

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
        scores = torch.sigmoid(output)[0]

    top_scores, top_indices = torch.topk(scores, 10)

    image = Draw.MolsToGridImage(
        [mol1, mol2],
        legends=[drug1_display.split(" [")[0], drug2_display.split(" [")[0]],
        molsPerRow=2,
        subImgSize=(350, 300),
    )

    result = "## Tahmin Edilen İlk 10 Yan Etki\n\n"
    result += "| Sıra | Yan Etki | Türkçe Karşılık | Açıklama | Kaynak | Model Skoru |\n"
    result += "|---|---|---|---|---|---|\n"

    for i, idx in enumerate(top_indices):
        label_id = mlb.classes_[idx.item()]
        effect_name = side_effect_descriptions.get(
            str(label_id), {}
        ).get("en_name", str(label_id))

        score = top_scores[i].item() * 100

        tr_name, description, source = get_side_effect_info(label_id, effect_name)

        result += (
            f"| {i + 1} | {effect_name} | {tr_name} | "
            f"{description} | {source} | %{score:.2f} |\n"
        )

    result += "\n---\n"
    result += "**Not:** Bu sistem tıbbi tanı veya tedavi önerisi değildir. "
    result += "Model skorları kesin klinik olasılık değil, TWOSIDES veri setinden öğrenilen örüntülere dayalı göreli tahmin skorlarıdır. "
    result += "Yan etki açıklamaları, önceden oluşturulmuş açıklama dosyasından alınmaktadır."

    return image, result


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi")

    gr.Markdown(
        "Bu arayüz, iki ilacın birlikte kullanımında ortaya çıkabilecek "
        "olası yan etkileri TWOSIDES veri seti üzerinde eğitilmiş GATv2 modeliyle tahmin eder."
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

    gr.Markdown("## Tahmin Sonuçları")
    output_text = gr.Markdown()

    button.click(
        predict_side_effects,
        inputs=[drug1, drug2],
        outputs=[output_image, output_text],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)