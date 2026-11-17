# 👗 Fashion Recommendation System

**Bir fotoğraf yükleyin, görsel olarak benzer moda ürünlerini saniyeler içinde bulun.**

[![Canlı demo](https://img.shields.io/badge/Canl%C4%B1%20demo-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://sibel-fashion-recommendation.streamlit.app/)
[![Veri seti](https://img.shields.io/badge/Veri%20seti-Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Sibeller/fashion-data)

Bu proje, derin öğrenme ile görüntülerden özellik çıkaran ve en benzer ürünleri öneren içerik tabanlı bir öneri sistemidir. Arayüz Streamlit ile hazırlanmıştır.

🔗 **Canlı demo:** https://sibel-fashion-recommendation.streamlit.app/
*(Uygulama bir süre kullanılmadığında uykuya geçer; ilk açılış birkaç dakika sürebilir.)*

---

## 🎯 Problem

E-ticarette kullanıcılar çoğu zaman beğendikleri bir ürünü kelimelerle tarif edemez. Bu sistem metin yerine **görsel benzerlik** ile arama yaparak bu sorunu çözer.

## ⚙️ Nasıl çalışır?

1. **Özellik çıkarımı:** ImageNet üzerinde önceden eğitilmiş **VGG16** modeli (son sınıflandırma katmanları olmadan) ve **Global Max Pooling** ile her görsel sayısal bir özellik vektörüne dönüştürülür. *(Transfer learning)*
2. **Benzerlik indeksi:** Tüm ürünlerin özellik vektörleri üzerinde **kosinüs mesafesi** ile çalışan bir **NearestNeighbors** (scikit-learn) modeli kurulur.
3. **Öneri:** Seçilen ya da yüklenen görselin vektörüne en yakın **5 ürün** listelenir. Sorgu ürününün kendisi sonuçlardan çıkarılır.

```
Görsel ──► VGG16 (ImageNet) ──► GlobalMaxPooling ──► Özellik vektörü ──► NearestNeighbors (cosine) ──► En benzer 5 ürün
```

## 🖥️ Uygulama özellikleri

Uygulamada iki sekme bulunur:

- **Listeden ürün seç:** Kaydırıcı ile bir ürün seçin, benzer 5 ürünü görün.
- **Fotoğraf yükle:** Kendi fotoğrafınızı yükleyin, sistem katalogdaki en benzer 5 ürünü göstersin.

## 📦 Veri

Ürün görselleri ve önceden hesaplanmış özellik vektörleri Hugging Face'te barındırılır: [`Sibeller/fashion-data`](https://huggingface.co/datasets/Sibeller/fashion-data). Uygulama ilk çalıştığında bu dosyaları otomatik olarak indirir ve önbelleğe alır.

Uygulama, Hugging Face'te görseli bulunan ürünlerle çalışır; görseli olmayan ürünler katalogdan otomatik olarak çıkarılır.

## 🛠️ Teknolojiler

`Python` · `TensorFlow / Keras (VGG16)` · `scikit-learn` · `Streamlit` · `Hugging Face Hub`

## 🚀 Kendi bilgisayarınızda çalıştırma

```bash
git clone https://github.com/ser2025/fashion-recommendation-system.git
cd fashion-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dosyalar

| Dosya | Açıklama |
|---|---|
| `app.py` | Streamlit uygulaması: veriyi indirir, modeli kurar, önerileri gösterir |
| `requirements.txt` | Gerekli Python paketleri |

## 📈 Geliştirme fikirleri

- [ ] Daha hafif ve hızlı modellerle (ResNet50, EfficientNet) karşılaştırma
- [ ] Büyük kataloglar için FAISS ile hızlı benzerlik araması
- [ ] Renk / kategori filtresi ekleme

## 👩‍💻 Geliştirici

**Sibel Er** · [LinkedIn](https://www.linkedin.com/in/sibel-e-b8166b162) · [Kaggle](https://www.kaggle.com/sibeler) · [Hugging Face](https://huggingface.co/Sibeller) · [Diğer projeler](https://github.com/ser2025/veri-bilimi-projeleri)
