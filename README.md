# GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi

Bu proje, iki ilacın birlikte kullanımında ortaya çıkabilecek olası yan etkileri tahmin etmek amacıyla geliştirilmiştir. Sistem, ilaçların kimyasal yapılarını kullanarak Graph Neural Network (GNN) tabanlı bir model ile çoklu yan etki tahmini yapmaktadır.

Projede TWOSIDES veri seti kullanılmıştır ve model PyTorch Geometric altyapısıyla geliştirilmiştir.

---

# Projenin Amacı

Bazı ilaçlar tek başına güvenli olsa bile birlikte kullanıldıklarında farklı yan etkiler oluşturabilir. Bu proje, ilaç çiftleri arasındaki bu etkileşimleri öğrenerek olası yan etkileri tahmin etmeyi amaçlamaktadır.

Sistem:
- İki ilacın kimyasal yapısını alır
- Moleküler graph yapısına dönüştürür
- GNN modeli ile analiz eder
- Olası yan etkiler için skor üretir
- En olası yan etkileri kullanıcıya sunar

---

# Kullanılan Teknolojiler

- Python
- PyTorch
- PyTorch Geometric
- RDKit
- Gradio
- TWOSIDES Dataset
- Ollama
- PubChemPy

---

# Kullanılan Veri Seti

Projede kullanılan veri seti:

## TWOSIDES

TWOSIDES veri seti:
- İlaç çiftleri
- İlaçların SMILES yapıları
- Gerçek yan etki etiketleri

bilgilerini içermektedir.

Toplam:
- 63.473 benzersiz ilaç çifti
- 1317 farklı yan etki sınıfı

kullanılmıştır.

---

# Model Mimarileri

Projede üç farklı Graph Neural Network mimarisi karşılaştırılmıştır:

| Model | Açıklama |
|---|---|
| GCN | Temel graph convolution yaklaşımı |
| GraphSAGE | Komşu düğüm bilgilerini örnekleyerek öğrenen yapı |
| GATv2 | Attention tabanlı graph neural network yapısı |

---

# Final Model

Yapılan deneyler sonucunda en yüksek performansı GraphSAGE modeli göstermiştir.

Bu nedenle final sistemde GraphSAGE tabanlı model kullanılmıştır.

## Final Model Performansı

| Metrik | Sonuç |
|---|---:|
| Test ROC-AUC | 0.8845 |
| Test PR-AUC | 0.3476 |

---

# Model Karşılaştırması

| Model | Test ROC-AUC | Test PR-AUC |
|---|---:|---:|
| GCN | 0.8781 | 0.3297 |
| GATv2 | 0.8797 | 0.3380 |
| GraphSAGE | **0.8845** | **0.3476** |

GraphSAGE modeli hem ROC-AUC hem PR-AUC metriklerinde en yüksek sonucu verdiği için final model olarak seçilmiştir.

---

# Sistem Nasıl Çalışır?

1. Kullanıcı iki ilaç seçer
2. İlaçların SMILES yapıları alınır
3. RDKit ile moleküler graph oluşturulur
4. Atom ve bağ özellikleri çıkarılır
5. GraphSAGE modeli graph embedding üretir
6. Model 1317 yan etki için skor üretir
7. En yüksek olasılıklı yan etkiler kullanıcıya gösterilir

---

# Kullanılan Özellikler

## Atom Özellikleri

Model aşağıdaki atom özelliklerini kullanmaktadır:

- Atom tipi
- Atom derecesi
- Hidrojen sayısı
- Valans bilgisi
- Formal charge
- Hybridization
- Aromatiklik
- Atom kütlesi

## Bağ Özellikleri

- Single bond
- Double bond
- Triple bond
- Aromatic bond
- Conjugated bond
- Ring bilgisi

---

# Uygulama Özellikleri

Arayüz üzerinde:

- İki ilaç seçilebilir
- İlaç eşleşmeleri filtrelenebilir
- Moleküler yapılar görüntülenebilir
- Top-K yan etki tahmini yapılabilir
- Gerçek veri seti etiketleri ile karşılaştırma yapılabilir
- Yan etkiler için Türkçe açıklamalar gösterilebilir
- İlaç bilgi kartları görüntülenebilir
- Model performans paneli görüntülenebilir

---

# Proje Yapısı

```text
twosides_project/
│
├── app.py
├── train.py
├── train_model_comparison.py
├── train_graphsage_final.py
│
├── model.py
├── model_gcn.py
├── model_graphsage.py
│
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
│   ├── graphsage_best.pth
│   ├── twosides_graphsage_model.pth
│   └── label_binarizer.pth
│
├── results/
│   ├── model_comparison.csv
│   ├── graphsage_final_results.csv
│   └── graphsage_final_history.csv
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

---

# Model Eğitimi

Final GraphSAGE modelini eğitmek için:

```bash
python train_graphsage_final.py
```

Model karşılaştırmalarını çalıştırmak için:

```bash
python train_model_comparison.py
```

---

# Not

Bu proje araştırma ve eğitim amaçlı geliştirilmiştir.

Üretilen sonuçlar:
- tıbbi tanı,
- tedavi önerisi,
- klinik karar sistemi

olarak kullanılmamalıdır.

---

# Geliştirici

Bilgisayar Mühendisliği Bitirme Projesi

GNN Tabanlı Polypharmacy Yan Etki Tahmin Sistemi