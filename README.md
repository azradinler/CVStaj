## Gereksinimler

- Python 3.8 veya üzeri
- PyTorch (https://pytorch.org/get-started/locally/)  
- torchvision
- matplotlib (Yanlış tahminlerin görselleşirilmesinde)

PyTorch ve torchvision paketlerini kurmak için:
[PyTorch Kurulum Sayfası](https://pytorch.org/get-started/locally/)

## Kullanım

- CNN-train.py çalıştırmak modeli eğitir ve models/best_model.pt olarak kaydeder. (1 tane örnek model bulunuyor (mnist_model.pt))
- CNN-test.py çalıştırmak eğitilmiş modeli alır ve test eder, doğruluğu yazar ve 10 yanış tahmini gösterir.
- Modelin yapısı simple_cnn.py içerisinde değiştirilebilir.

 Örnek Test Çıktısı:
 
<img width="1784" height="866" alt="image" src="https://github.com/user-attachments/assets/e9f9f46e-cf3c-4b34-bbac-eeb464cc1f39" />


