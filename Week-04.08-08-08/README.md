# TC Kimlik Kartından Bilgi Çıkarımı

---

## Hakkında
Tesseract OCR kullanarak TC Kimlik kartı görsellerinden tc no, ad, soyad, doğum tarihi bilgileri çıkarmaya yarayan basit bir uygulama.



### Gereksinimler

Docker

### Kurulum

Docker Desktop uygulamasını çalıştırın.

**Docker ile build edin:**
```bash
docker build -t app .
```

**Containeri başlatın:**
```bash
docker run -p 5000:5000 app
```

**Tarayıcınızda açın:**
```
http://127.0.0.1:5000
```

### Screenshots
<img width="915" height="1230" alt="Ekran görüntüsü 2025-08-08 125510" src="https://github.com/user-attachments/assets/273db176-29e6-4044-8b91-82e2eb737aa4" />

