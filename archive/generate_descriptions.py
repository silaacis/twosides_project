import json
import time
from deep_translator import GoogleTranslator
from tdc.utils import get_label_map


OUTPUT_FILE = "side_effect_descriptions.json"

translator = GoogleTranslator(source="en", target="tr")


def create_safe_description(effect_name, tr_name):
    return (
        f"{tr_name}, TWOSIDES veri setinde ilaç kombinasyonlarıyla ilişkili "
        "olarak bildirilen klinik bir yan etkidir. Bu açıklama tıbbi tanı veya "
        "tedavi önerisi değildir."
    )


def main():
    print("TWOSIDES yan etki etiketleri alınıyor...")

    label_map = get_label_map(
        name="TWOSIDES",
        task="DDI",
        name_column="Side Effect Name",
    )

    descriptions = {}

    for label_id, effect_name in label_map.items():
        effect_name = str(effect_name)

        try:
            tr_name = translator.translate(effect_name).title()
        except Exception:
            tr_name = effect_name

        descriptions[str(label_id)] = {
            "en_name": effect_name,
            "tr_name": tr_name,
            "description": create_safe_description(effect_name, tr_name),
            "source": "Otomatik açıklama"
        }

        time.sleep(0.05)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(descriptions, file, ensure_ascii=False, indent=4)

    print(f"Açıklama dosyası oluşturuldu: {OUTPUT_FILE}")
    print(f"Toplam yan etki sayısı: {len(descriptions)}")


if __name__ == "__main__":
    main()