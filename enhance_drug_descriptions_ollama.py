import json
import time
import requests

INPUT_FILE = "drug_list_named.json"
OUTPUT_FILE = "drug_descriptions_enhanced.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def ask_ollama(drug_name, cid, smiles):
    prompt = f"""
Sen ilaç etken maddelerini sade Türkçe ile açıklayan bir asistansın.

İlaç/Bileşik:
{drug_name}

PubChem CID:
{cid}

SMILES:
{smiles}

Kurallar:
- Türkçe yaz.
- En fazla 2 cümle kur.
- İlacın genel kullanım amacını sade şekilde açıkla.
- Marka adı verme.
- Kimyasal jargon kullanma.
- Doktor tavsiyesi verme.
- Eğer emin değilsen:
"Bu bileşik hakkında güvenilir kısa kullanım bilgisi bulunamadı." yaz.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()
    return response.json().get("response", "").strip()


def is_bad_drug_name(name):
    if not name:
        return True

    lowered = name.lower().strip()

    suspicious_tokens = [
        "stk",
        "mls",
        "cid",
        "chembl",
        "zinc",
        "pubchem",
    ]

    if any(token in lowered for token in suspicious_tokens):
        return True

    digit_count = sum(char.isdigit() for char in name)

    if digit_count >= len(name) * 0.4:
        return True

    if "-" in name and digit_count >= 4:
        return True

    return False


def is_bad_description(text):
    if not text or len(text.strip()) < 25:
        return True

    lowered = text.lower()

    banned = [
        "doktor",
        "danış",
        "tedavi önerisi",
        "kullanım talimatı",
        "üzgünüm",
        "bilmiyorum",
    ]

    if any(word in lowered for word in banned):
        return True

    words = lowered.split()

    for word in set(words):
        if words.count(word) >= 4:
            return True

    return False


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        drugs = json.load(file)

    descriptions = {}

    total = len(drugs)
    print(f"Toplam ilaç/bileşik: {total}")

    for idx, (cid, item) in enumerate(drugs.items(), start=1):
        drug_name = item.get("name", cid)
        smiles = item.get("smiles", "")

        print(f"[{idx}/{total}] {drug_name} ({cid}) işleniyor...")

        try:
            if is_bad_drug_name(drug_name):
                raise ValueError("Şüpheli veya anlamsız ilaç adı.")

            description = ask_ollama(
                drug_name=drug_name,
                cid=cid,
                smiles=smiles,
            )

            if is_bad_description(description):
                raise ValueError("Açıklama kalite kontrolünden geçmedi.")

            descriptions[cid] = {
                "cid": cid,
                "name": drug_name,
                "smiles": smiles,
                "description": description,
                "source": "Ollama Gemma3",
            }

        except Exception as error:
            print(f"  Uyarı: {error}")

            descriptions[cid] = {
                "cid": cid,
                "name": drug_name,
                "smiles": smiles,
                "description": "Bu bileşik hakkında güvenilir kısa kullanım bilgisi bulunamadı.",
                "source": "Varsayılan",
            }

        if idx % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
                json.dump(descriptions, file, ensure_ascii=False, indent=4)

        time.sleep(0.2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(descriptions, file, ensure_ascii=False, indent=4)

    print(f"\nTamamlandı: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()