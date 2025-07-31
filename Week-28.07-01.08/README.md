# OpenClip ile Sadece Anlatarak Fotoğraflarınızı Bulun

---

## Hakkında

Bilgisayarınızda bulunan fotoğraflarınız arasında Open Clip modeli sayesinde metinsel ifadeler ile arama yapmanızı sağlayan basit arayüze sahip basit bir uygulama.

### Özellikler

- OpenCLIP modeli ile metin-görsel eşleştirmesi.
- Streamlit ile web arayüzü
- Daha hızlı arama için verileri kaydetme imkanı
---

## Kurulum

### Gereksinimler

- PyTorch [PyTorch Kurulum](https://pytorch.org/get-started/locally/)

**Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

**Uygulamayı çalıştırın:**
```bash
streamlit run imgsearch.py
```

**Tarayıcınızda açın:**
```
http://localhost:8501
```

## Kullanım Kılavuzu

1. **Klasör Seçimi**
   - Sol panelden fotoğraf klasörünüzü seçin
   - Sistem otomatik olarak desteklenen formatları tarayacak
<img width="1299" height="825" alt="image" src="https://github.com/user-attachments/assets/5f80ab20-2fc2-4964-a0cd-8bdf8dc62554" />

---

2. **Arama Modu ve Sonuç Miktarı Seçimi**
   - **Gerçek Zamanlı**: Anında kaydetmeden arama (yavaş ama güncel)
   - **İndekslenmiş**: Hızlı arama (önce indeksleme gerekli)
  <img width="478" height="800" alt="image" src="https://github.com/user-attachments/assets/fea761ce-6c87-40bd-8006-5f2818e60452" />
  
---

3. **İndeksleme (İsteğe Bağlı)**
   - "Klasörü İndeksle" butonuna tıklayın
   - Sonraki aramalar daha hızlı olacak
    <img width="600" height="935" alt="image" src="https://github.com/user-attachments/assets/4f7e7d66-84d6-43b0-bb2d-1bb4943ce28f" />
    
---

4. **Arama Yapma**
   - Arama kutusuna sorgunuzu yazın (İngilizce)
   - "Ara" butonuna tıklayın
   - Sonuçları görüntüleyin
     <img width="2488" height="1159" alt="image" src="https://github.com/user-attachments/assets/9a9f6acd-dcda-4e69-9124-1e1ae1eb0927" />

