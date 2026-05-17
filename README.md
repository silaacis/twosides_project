# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi

Bu proje, iki ilacın birlikte kullanımında ortaya çıkabilecek olası yan etkileri tahmin etmek amacıyla geliştirilmiştir. Sistem, ilaçların kimyasal yapılarını kullanarak Graph Neural Network (GNN) tabanlı bir model ile yan etki tahmini yapmaktadır.

Projede TWOSIDES veri seti kullanılmıştır ve model PyTorch Geometric altyapısıyla geliştirilmiştir.

---

# Projenin Amacı

Bazı ilaçlar tek başına güvenli olsa bile birlikte kullanıldıklarında farklı yan etkilere neden olabilir. Bu proje, ilaç çiftleri arasındaki bu etkileşimleri öğrenerek olası yan etkileri tahmin etmeyi amaçlamaktadır.

Sistem:
- İki ilacın kimyasal yapısını alır
- Moleküler graph yapısına dönüştürür
- GNN modeli ile analiz eder
- Olası yan etkileri skorlayarak kullanıcıya gösterir

---

# Kullanılan Teknolojiler

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Gradio
- TWOSIDES Dataset
- Ollama (Yerel AI açıklama sistemi)

---

# Model Yapısı

Projede Siamese GATv2 tabanlı bir Graph Neural Network modeli kullanılmıştır.

Model:
1. İki ilacın moleküler graph yapısını oluşturur
2. Atom ve bağ özelliklerini analiz eder
3. Her ilaç için embedding üretir
4. İki ilacın birlikte oluşturduğu yapıyı öğrenir
5. Olası yan etkiler için skor üretir

Toplam:
- 1317 farklı yan etki sınıfı
- Çok etiketli (multi-label) tahmin sistemi

---

# Veri Seti

Projede kullanılan veri seti:

TWOSIDES

Veri seti:
- İlaç çiftleri
- İlaçların SMILES yapıları
- Gerçek yan etki etiketleri

bilgilerini içermektedir.

---

# Eğitim Sonuçları

Model eğitiminden elde edilen sonuçlar:

| Metrik | Sonuç |
|---|---|
| Test ROC-AUC | 0.8828 |
| Test PR-AUC | 0.3437 |

Bu sonuçlar modelin yan etki örüntülerini başarılı şekilde öğrenebildiğini göstermektedir.

---

# Uygulama Özellikleri

Arayüz üzerinde:

- İki ilaç seçilebilir
- Moleküler yapılar görüntülenebilir
- Top-K yan etki tahmini yapılabilir
- Gerçek veri seti etiketleri ile karşılaştırma yapılabilir
- Yan etkiler için Türkçe açıklamalar gösterilebilir
- İlaçlar hakkında kısa kullanım bilgileri görüntülenebilir

---

# Proje Yapısı

```text
twosides_project/
│
├── app.py
├── train.py
├── model.py
├── graph_utils.py
├── requirements.txt
│
├── side_effect_descriptions_enhanced.json
├── drug_descriptions_enhanced.json
│
├── enhance_side_effect_descriptions_ollama.py
├── enhance_drug_descriptions_ollama.py
│
├── models/
│   ├── twosides_gatv2_model.pth
│   ├── twosides_gatv2_best_model.pth
│   └── label_binarizer.pth
│
├── results/
│   ├── training_history.csv
│   └── final_test_results.csv
│
└── README.md
```

---

# Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

---

# Uygulamayı Çalıştırma

Arayüzü başlatmak için:

```bash
python app.py
```

Model eğitimi yapmak için:

```bash
python train.py
```

---

# Not

Bu proje araştırma ve eğitim amaçlı geliştirilmiştir. Üretilen sonuçlar tıbbi tanı veya tedavi önerisi değildir.

---

# Geliştirici

Bilgisayar Mühendisliği Bitirme Projesi
GNN Tabanlı Polypharmacy Yan Etki Tahmini