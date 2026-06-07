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
from model_graphsage_fp import SiameseGraphSAGEFingerprint
from fingerprint_utils import smiles_to_morgan_fingerprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SIDE_EFFECT_DESCRIPTION_FILE = "side_effect_descriptions_enhanced.json"
DRUG_DESCRIPTION_FILE = "drug_descriptions_enhanced.json"

TEST_ROC_AUC = 0.8922
TEST_PR_AUC = 0.3668
NUM_SIDE_EFFECT_CLASSES = 1317


def load_json_file(file_path, warning_message):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    print(warning_message)
    return {}


side_effect_descriptions = load_json_file(
    SIDE_EFFECT_DESCRIPTION_FILE,
    "Uyarı: side_effect_descriptions_enhanced.json bulunamadı.",
)

drug_descriptions = load_json_file(
    DRUG_DESCRIPTION_FILE,
    "Uyarı: drug_descriptions_enhanced.json bulunamadı.",
)


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
    key = str(label_id)

    side_effect_name = key

    try:
        int_key = int(key)
        side_effect_name = label_map.get(int_key, key)
    except Exception:
        side_effect_name = label_map.get(key, key)

    possible_keys = [
        key,
        side_effect_name,
        str(side_effect_name),
        str(side_effect_name).lower(),
    ]

    for lookup_key in possible_keys:
        if lookup_key in side_effect_descriptions:
            item = side_effect_descriptions[lookup_key]
            return {
                "en_name": item.get("en_name", side_effect_name),
                "tr_name": item.get("tr_name", item.get("en_name", side_effect_name)),
                "description": item.get("description", "Açıklama bulunamadı."),
                "source": item.get("source", "Açıklama dosyası"),
            }

    return {
        "en_name": side_effect_name,
        "tr_name": side_effect_name,
        "description": "Bu yan etki için açıklama bulunamadı.",
        "source": "Varsayılan",
    }
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


def get_compound_from_cid(cid):
    try:
        pure_cid = int(str(cid).replace("CID", ""))
        return pcp.Compound.from_cid(pure_cid)
    except Exception:
        return None


def get_drug_name(cid):
    try:
        title_name = get_pubchem_record_title(cid)
        if title_name:
            return title_name

        compound = get_compound_from_cid(cid)

        if compound is None:
            return f"Bilinmeyen İlaç [{cid}]"

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


def get_drug_usage_description(cid):
    item = drug_descriptions.get(cid, {})

    description = item.get(
        "description",
        "Bu bileşik hakkında güvenilir kısa kullanım bilgisi bulunamadı.",
    )

    source = item.get("source", "Varsayılan")
    name = item.get("name", "")

    return name, description, source


def get_drug_info_markdown(display_name, cid, smiles):
    compound = get_compound_from_cid(cid)
    name = display_name.split(" [")[0]

    stored_name, drug_description, drug_description_source = get_drug_usage_description(cid)

    if stored_name and not stored_name.upper().startswith("CID"):
        shown_name = stored_name
    else:
        shown_name = name

    if compound is None:
        return (
            f"### {shown_name}\n\n"
            f"- **PubChem CID:** {cid}\n"
            f"- **Kısa Açıklama / Kullanım Bilgisi:** {drug_description}\n"
            f"- **Açıklama Kaynağı:** {drug_description_source}\n"
            f"- **SMILES:** `{smiles}`\n"
            f"- PubChem üzerinden ayrıntılı kimyasal bilgi alınamadı.\n"
        )

    formula = compound.molecular_formula or "Bilinmiyor"
    weight = compound.molecular_weight or "Bilinmiyor"
    iupac = compound.iupac_name or "Bilinmiyor"

    synonyms = compound.synonyms[:5] if compound.synonyms else []
    synonyms_text = ", ".join(synonyms) if synonyms else "Bilinmiyor"

    return (
        f"### {shown_name}\n\n"
        f"- **PubChem CID:** {cid}\n"
        f"- **Kısa Açıklama / Kullanım Bilgisi:** {drug_description}\n"
        f"- **Açıklama Kaynağı:** {drug_description_source}\n"
        f"- **Moleküler Formül:** {formula}\n"
        f"- **Molekül Ağırlığı:** {weight}\n"
        f"- **IUPAC Adı:** {iupac}\n"
        f"- **Diğer Adlar:** {synonyms_text}\n"
        f"- **SMILES:** `{smiles}`\n"
    )


def get_score_level(score):
    if score >= 45:
        return "Yüksek"
    if score >= 30:
        return "Orta"
    return "Düşük"


print(f"Kullanılan cihaz: {device}")
print("TWOSIDES verisi yükleniyor...")

data = DDI(name="TWOSIDES")



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
    d1 = str(row["Drug1_ID"])
    d2 = str(row["Drug2_ID"])
    labels = set(str(label) for label in row["Y"])

    true_label_dict[(d1, d2)] = labels
    true_label_dict[(d2, d1)] = labels


print("İlaç isimleri hazırlanıyor...")

drug_dict = {}
display_to_id = {}
id_to_display = {}
all_drugs = {}

for _, row in grouped_df.iterrows():
    all_drugs[str(row["Drug1_ID"])] = row["Drug1"]
    all_drugs[str(row["Drug2_ID"])] = row["Drug2"]

for cid, smiles in all_drugs.items():
    drug_name = get_drug_name(cid)

    if drug_name.startswith("Bilinmeyen İlaç"):
        display_name = drug_name
    else:
        display_name = f"{drug_name} [{cid}]"

    drug_dict[cid] = smiles
    display_to_id[display_name] = cid
    id_to_display[cid] = display_name

drug_list = sorted(list(display_to_id.keys()))

print("İlaç eşleşme haritası hazırlanıyor...")

drug_pair_map = {}

for _, row in grouped_df.iterrows():
    d1 = str(row["Drug1_ID"])
    d2 = str(row["Drug2_ID"])

    display1 = id_to_display.get(d1)
    display2 = id_to_display.get(d2)

    if display1 and display2:
        drug_pair_map.setdefault(display1, set()).add(display2)
        drug_pair_map.setdefault(display2, set()).add(display1)


def update_second_drug(selected_drug):
    if selected_drug is None:
        return gr.Dropdown(choices=[], value=None)

    possible_pairs = sorted(list(drug_pair_map.get(selected_drug, [])))

    return gr.Dropdown(
        choices=possible_pairs,
        value=None,
        label="İkinci İlaç",
        filterable=True,
    )


model = SiameseGraphSAGEFingerprint(
    num_node_features=80,
    num_edge_features=6,
    hidden_dim=128,
    num_classes=num_classes,
    fingerprint_dim=512,
).to(device)

model.load_state_dict(
    torch.load(
        "models/graphsage_fp_best.pth",
        map_location=device,
    )
)

model.eval()

print("GraphSAGE + Morgan Fingerprint modeli başarıyla yüklendi.")


def predict_side_effects(drug1_display, drug2_display, top_k):
    empty_df = pd.DataFrame()

    if drug1_display is None or drug2_display is None:
        return None, "", "", "Lütfen iki ilaç seçiniz.", empty_df, empty_df

    drug1_id = display_to_id[drug1_display]
    drug2_id = display_to_id[drug2_display]

    if drug1_id == drug2_id:
        return None, "", "", "Lütfen iki farklı ilaç seçiniz.", empty_df, empty_df

    top_k = int(top_k)

    smiles1 = drug_dict[drug1_id]
    smiles2 = drug_dict[drug2_id]

    drug1_info = get_drug_info_markdown(drug1_display, drug1_id, smiles1)
    drug2_info = get_drug_info_markdown(drug2_display, drug2_id, smiles2)

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    if mol1 is None or mol2 is None:
        return None, drug1_info, drug2_info, "Molekül yapısı okunamadı.", empty_df, empty_df

    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return None, drug1_info, drug2_info, "Graf dönüşümü başarısız oldu.", empty_df, empty_df

    batch1 = Batch.from_data_list([g1]).to(device)
    batch2 = Batch.from_data_list([g2]).to(device)

    fp1 = smiles_to_morgan_fingerprint(smiles1).unsqueeze(0).to(device)
    fp2 = smiles_to_morgan_fingerprint(smiles2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(batch1, batch2, fp1, fp2)
        scores = torch.sigmoid(output)[0]

    top_scores, top_indices = torch.topk(scores, top_k)

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
                "Skor Seviyesi": get_score_level(score),
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

    precision_at_k = matched_count / top_k if top_k > 0 else 0

    summary = (
        f"### Değerlendirme Özeti\n\n"
        f"- **Top-{top_k} eşleşme:** {matched_count} / {top_k}\n"
        f"- **Precision@{top_k}:** {precision_at_k:.2f}\n"
        f"- **Veri setindeki gerçek yan etki sayısı:** {len(true_labels)}\n\n"
        f"Bu karşılaştırma yalnızca TWOSIDES veri setinde kayıtlı ilaç çiftleri için yapılmaktadır. "
        f"Model skorları kesin klinik olasılık değil, veri setinden öğrenilen örüntülere dayalı göreli tahmin skorlarıdır."
    )

    return image, drug1_info, drug2_info, summary, prediction_df, true_df


performance_panel = f"""
## Model Performans Özeti

| Metrik | Değer |
|---|---:|
| Test ROC-AUC | {TEST_ROC_AUC:.4f} |
| Test PR-AUC | {TEST_PR_AUC:.4f} |
| Yan etki sınıfı sayısı | {NUM_SIDE_EFFECT_CLASSES} |
| Model | GraphSAGE + Morgan Fingerprint |
| Veri seti | TWOSIDES |

"""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi")

    gr.Markdown(
        "Bu arayüz, iki ilacın birlikte kullanımında ortaya çıkabilecek olası yan etkileri "
        "TWOSIDES veri seti üzerinde eğitilmiş GraphSAGE + Morgan Fingerprint tabanlı GNN modeliyle tahmin eder."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Girdi Paneli")

            drug1 = gr.Dropdown(
                choices=drug_list,
                label="Birinci İlaç",
                filterable=True,
            )

            drug2 = gr.Dropdown(
                choices=[],
                label="İkinci İlaç",
                filterable=True,
            )

            top_k = gr.Radio(
                choices=[5, 10, 20],
                value=10,
                label="Gösterilecek Tahmin Sayısı (Top-K)",
            )

            button = gr.Button("Yan Etki Tahmini Yap", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown(performance_panel)
            output_image = gr.Image(label="Moleküler Yapılar")

    gr.Markdown(
        "**Not:** Veri setinde ilaçlar genellikle marka adıyla değil, "
        "etken madde veya PubChem bileşik adıyla temsil edilmektedir."
    )

    gr.Markdown("## Seçilen İlaç Bilgileri")

    with gr.Row():
        with gr.Column(scale=1):
            drug1_info_box = gr.Markdown()

        with gr.Column(scale=1):
            drug2_info_box = gr.Markdown()

    output_summary = gr.Markdown()

    gr.Markdown("## Modelin Tahmin Ettiği Yan Etkiler")

    prediction_table = gr.Dataframe(
        interactive=False,
        wrap=True,
    )

    gr.Markdown("## TWOSIDES Veri Setindeki Gerçek Yan Etkiler")

    true_table = gr.Dataframe(
        interactive=False,
        wrap=True,
    )

    gr.Markdown(
        "**Not:** Bu sistem tıbbi tanı veya tedavi önerisi değildir. "
        "Sonuçlar araştırma/prototip amaçlıdır. İlaç bilgi kartlarındaki açıklamalar "
        "kullanıcıya genel fikir vermek amacıyla oluşturulmuştur."
    )

    drug1.change(
        fn=update_second_drug,
        inputs=drug1,
        outputs=drug2,
    )

    button.click(
        predict_side_effects,
        inputs=[drug1, drug2, top_k],
        outputs=[
            output_image,
            drug1_info_box,
            drug2_info_box,
            output_summary,
            prediction_table,
            true_table,
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=True)