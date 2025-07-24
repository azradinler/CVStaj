\# Hakkında

Bu proje, video veya görsellerde insan tespiti, takibi ve temel poz (duruş) tahmini yapabilen bir Python uygulamasıdır. Ultralytics YOLOv8 ve pose modelleri ile çalışır. İnsanları tespit edip her bir kişiyi takip eder, ayrıca kişinin ayakta, oturuyor, yürüyor veya koşuyor gibi temel pozlarını tahmin etmeye çalışır



\## Temel Özellikler

\- İnsan tespiti ve takibi

\- Kişi başına ID atama

\- Temel poz (ayakta, oturuyor, yürüyor, koşuyor) tahmini

\- Sonuçların ekranda kutu ve anahtar noktalarla gösterimi



\## Ana Dosyalar

\- `humantrack.py`: Sadece insan tespiti ve takibi

\- `posetrack.py`: İnsan tespiti, takibi ve poz tahmini

\- `persontracker.py`: İnsan + poz takipçi sınıflarını içeren dosya

\- 



\## Gereksinimler



\- PyTorch 

\[PyTorch Kurulum](https://pytorch.org/get-started/locally/)



\## Kullanım

Sadece insan takibi için:

humantrack.py içinde input video kısmını değiştirip çalıştırın, ekranda oynatacaktır



Ekran Görüntüleri:

<img width="2133" height="1201" alt="image" src="https://github.com/user-attachments/assets/6af8d157-60d6-44b6-957e-6b0c532422e2" />









İnsan takibi + poz tahmini için:

posetrack.py içinde input video kısmını değiştirip çalıştırın



Ekran Görüntüleri:

<img width="2115" height="1183" alt="image" src="https://github.com/user-attachments/assets/fb6ccb72-c53d-43df-bc02-6e141e2bb901" />









