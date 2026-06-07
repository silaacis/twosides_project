import json
import time
import requests

INPUT_FILE = "side_effect_descriptions.json"
OUTPUT_FILE = "side_effect_descriptions_enhanced.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def ask_ollama(effect_en, effect_tr):
    prompt = f"""
Sen tıbbi terimleri sade Türkçeyle açıklayan bir asistansın.

Yan etki İngilizce adı: {effect_en}
Yan etki Türkçe adı: {effect_tr}

Görev:
Bu yan etkiyi Türkçe olarak yalnızca 1 cümleyle açıkla.
Tıbbi tavsiye verme.
"Doktora danışın" gibi yönlendirme yazma.
Sadece tanım yap.
Cevap kısa, net ve anlaşılır olsun.
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
    data = response.json()

    return data.get("response", "").strip()


def is_bad_description(text):
    if not text:
        return True

    if len(text) < 25:
        return True

    lowered = text.lower()

    bad_phrases = [
        "doktor",
        "tavsiye",
        "danış",
        "bilmiyorum",
        "emin değilim",
        "üzgünüm",
        "ben bir",
    ]

    if any(phrase in lowered for phrase in bad_phrases):
        return True

    words = lowered.split()

    for word in set(words):
        if words.count(word) >= 4:
            return True

    return False


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        descriptions = json.load(file)

    enhanced = {}

    total = len(descriptions)

    for index, (label_id, item) in enumerate(descriptions.items(), start=1):
        effect_en = item.get("en_name", "")
        effect_tr = item.get("tr_name", effect_en)

        print(f"[{index}/{total}] {effect_en} açıklanıyor...")

        try:
            new_description = ask_ollama(effect_en, effect_tr)

            if is_bad_description(new_description):
                raise ValueError("Açıklama kalite kontrolünden geçmedi.")

            item["description"] = new_description
            item["source"] = "Ollama Gemma3"

        except Exception as error:
            print(f"  Uyarı: {effect_en} için eski açıklama korundu. Sebep: {error}")

        enhanced[label_id] = item

        if index % 10 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
                json.dump(enhanced, file, ensure_ascii=False, indent=4)

        time.sleep(0.2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(enhanced, file, ensure_ascii=False, indent=4)

    print(f"Tamamlandı: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()