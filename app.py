import os
import json
import torch
import gradio as gr
import pubchempy as pcp

from transformers import pipeline
from deep_translator import GoogleTranslator
from tdc.multi_pred import DDI
from tdc.utils import get_label_map
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Batch

from graph_utils import smiles_to_graph
from model import SiameseGATv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translator = GoogleTranslator(source="en", target="tr")

print("AI açıklama modeli yükleniyor...")

explanation_generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=0 if torch.cuda.is_available() else -1,
)

CACHE_FILE = "side_effect_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=4)


side_effect_cache = load_cache()


def get_side_effect_explanation(effect_name):
    key = effect_name.lower().strip()

    if key in side_effect_cache:
        return side_effect_cache[key]["tr_name"], side_effect_cache[key]["description"]

    try:
        tr_name = translator.translate(effect_name)
        tr_name = tr_name.title()
    except Exception:
        tr_name = effect_name

    prompt = (
        "Explain this medical side effect in one simple English sentence. "
        "Do not give medical advice. "
        f"Side effect: {effect_name}. Explanation:"
    )

    try:
        ai_output = explanation_generator(
            prompt,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=50256,
        )

        english_description = ai_output[0]["generated_text"].replace(prompt, "").strip()

        if len(english_description) < 20:
            english_description = (
                f"{effect_name} is a clinical side effect that may be reported "
                "in relation to medication use."
            )

        try:
            description = translator.translate(english_description)
        except Exception:
            description = english_description

    except Exception:
        description = (
            f"{tr_name}, TWOSIDES veri setinde ilaç kombinasyonlarıyla ilişkili "
            "olarak bildirilen klinik bir yan etkidir."
        )

    side_effect_cache[key] = {
        "tr_name": tr_name,
        "description": description,
    }

    save_cache(side_effect_cache)

    return tr_name, description


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

label_map = get_label_map(
    name="TWOSIDES",
    task="DDI",
    name_column="Side Effect Name",
)

print("İlaç isimleri hazırlanıyor...")

drug_dict = {}
display_to_id = {}
all_drugs = {}

for _, row in grouped_df.iterrows():
    all_drugs[row["Drug1_ID"]] = row["Drug1"]
    all_drugs[row["Drug2_ID"]] = row["Drug2"]


def get_drug_name(cid):
    try:
        pure_cid = int(str(cid).replace("CID", ""))
        compound = pcp.Compound.from_cid(pure_cid)

        if compound.synonyms:
            clean_names = [
                name
                for name in compound.synonyms
                if len(name) < 40 and not any(char.isdigit() for char in name)
            ]

            if clean_names:
                return clean_names[0].title()

            return compound.synonyms[0].title()

    except Exception:
        pass

    return cid


for cid, smiles in all_drugs.items():
    drug_name = get_drug_name(cid)
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
        probs = torch.sigmoid(output)[0]

    top_probs, top_indices = torch.topk(probs, 10)

    image = Draw.MolsToGridImage(
        [mol1, mol2],
        legends=[drug1_display.split(" [")[0], drug2_display.split(" [")[0]],
        molsPerRow=2,
        subImgSize=(350, 300),
    )

    result = "## Tahmin Edilen İlk 10 Yan Etki\n\n"
    result += "| Sıra | Yan Etki | Türkçe Karşılık | AI Açıklaması | Olasılık |\n"
    result += "|---|---|---|---|---|\n"

    for i, idx in enumerate(top_indices):
        label_id = mlb.classes_[idx.item()]
        side_effect_name = label_map.get(label_id, str(label_id))
        probability = top_probs[i].item() * 100

        tr_name, description = get_side_effect_explanation(side_effect_name)

        result += (
            f"| {i + 1} | {side_effect_name} | {tr_name} | "
            f"{description} | %{probability:.2f} |\n"
        )

    result += "\n---\n"
    result += "**Not:** Bu sistem tıbbi tanı veya tedavi önerisi değildir. "
    result += "Model yalnızca TWOSIDES veri setindeki örüntülere dayalı tahmin üretir. "
    result += "Yan etki açıklamaları, kullanıcıya anlaşılır bilgi sunmak amacıyla AI destekli açıklama modülüyle oluşturulmuştur."

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