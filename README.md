# 👗 Fashion Recommendation System

**Bir fotoğraf yükleyin, görsel olarak benzer moda ürünlerini saniyeler içinde bulun.**

Bu proje, derin öğrenme ile görüntülerden özellik çıkaran ve en benzer ürünleri öneren içerik tabanlı bir öneri sistemidir. Arayüz Streamlit ile hazırlanmıştır.

<!-- Uygulamanın ekran görüntüsünü veya kısa bir GIF'ini buraya ekleyin -->
<!-- ![Uygulama ekran görüntüsü](screenshot.png) -->

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

- **Katalogdan seç:** Kaydırıcı ile bir ürün seçin, benzer 5 ürünü görün.
- **Fotoğraf yükle:** Kendi fotoğrafınızı yükleyin, sistem katalogdaki en benzer 5 ürünü göstersin.

## 📦 Veri

Ürün görselleri ve önceden hesaplanmış özellik vektörleri Hugging Face üzerinde barındırılır: [`Sibeller/fashion-data`](https://huggingface.co/datasets/Sibeller/fashion-data). Uygulama ilk çalıştığında bu dosyaları otomatik olarak indirir ve önbelleğe alır.

## 🛠️ Teknolojiler

`Python` · `TensorFlow / Keras (VGG16)` · `scikit-learn` · `Streamlit` · `Hugging Face Hub`

## 🚀 Kurulum ve çalıştırma

```bash
git clone https://github.com/ser2025/fashion-recommendation-system.git
cd fashion-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

<!-- Canlı demo varsa: 🔗 **Canlı demo:** [buraya bağlantı](https://...) -->

## 📈 Geliştirme fikirleri

- [ ] Daha hafif ve hızlı modellerle (ResNet50, EfficientNet) karşılaştırma
- [ ] Büyük kataloglar için FAISS ile hızlı benzerlik araması
- [ ] Renk / kategori filtresi ekleme
- [ ] Streamlit Community Cloud veya Hugging Face Spaces'te canlı demo

## 👩‍💻 Geliştirici

**Sibel Er** · [LinkedIn](https://www.linkedin.com/in/sibeler) · [Kaggle](https://www.kaggle.com/sibeler) · [GitHub](https://github.com/ser2025)
