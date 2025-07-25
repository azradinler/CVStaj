# CVStaj
Görüntü işleme alanında yapılan çalışmalar genellikle bazı temel görevlere dayanır. 
Bunlardan en yaygın olanları sınıflandırma (classification), nesne tespiti (detection) ve 
segmentasyon (segmentation) görevleridir. Her bir görev, görüntüdeki bilgiyi farklı şekilde 
yorumlamayı amaçlar. Bu görevlerin farklarını anlamak, hangi problemde hangi yaklaşımın 
kullanılacağına karar vermek açısından oldukça önemlidir. 
�
�
 1. Sınıflandırma (Classification) 
Sınıflandırma, bir görüntünün neye ait olduğunu belirlemeye yönelik bir işlemdir. Örneğin, 
elimizde bir hayvan fotoğrafı varsa ve bu fotoğrafın bir "kedi" mi yoksa "köpek" mi olduğunu 
anlamaya çalışıyorsak, bu bir sınıflandırma problemidir. 
● Bu görevde görüntü bir bütün olarak ele alınır. 
● Sistem, görüntüyü analiz eder ve önceden tanımlı sınıflardan birini seçer. 
● Örneğin: Bir görüntü → “Uçak” olarak etiketlenir. 
✅
 Uygulama Alanları: 
● El yazısı rakam tanıma 
● Tıbbi görüntülerde hastalık teşhisi 
● Sosyal medya fotoğraflarının içerik sınıflandırması 
�
�
 2. Nesne Tespiti (Detection) 
Nesne tespiti, bir görüntüde birden fazla nesne olup olmadığını ve bunların nerede 
bulunduğunu belirleme işidir. Sistem, hem her nesnenin ne olduğunu (etiket), hem de 
görüntü üzerindeki konumunu (genellikle dikdörtgen kutularla) belirtir. 
● Örneğin: Bir sokak görüntüsünde hem “insan” hem “araba” tespit edilip, her biri 
kutularla işaretlenir. 
● 
görevde konum bilgisi çok önemlidir. 
Bu 
✅
 Uygulama Alanları: 
● Güvenlik kameralarında kişi takibi 
● Otonom araçlarda yayaların tespiti 
● Barkod veya nesne tanıma sistemleri 
�
�
 3. Segmentasyon (Segmentation) 
Segmentasyon, bir görüntüdeki her bir pikselin hangi nesneye ait olduğunu belirleme 
işlemidir. Yani, sadece nesnelerin yerini değil, tam sınırlarını da öğrenmek isteriz. 
Segmentasyon sayesinde görüntü çok daha detaylı şekilde analiz edilebilir. 
Bu görev ikiye ayrılır: 
● Semantic Segmentation: Aynı türdeki tüm nesneler tek bir sınıfa aittir (örneğin tüm 
ağaçlar). 
● Instance Segmentation: Her nesne ayrı ayrı tanımlanır (her ağaç ayrı olarak). 
● Segmentasyonun çıktısı genellikle görüntüyle aynı boyutta olup, her piksel bir sınıfla 
etiketlenmiştir. 
✅
 Uygulama Alanları: 
● Tıbbi görüntülerde tümör bölgesi tespiti 
● Uydu görüntülerinde bina/alan ayrımı 
● Yol çizgilerinin ve şeritlerin tespiti 
YOLO (You Only Look Once), görüntüdeki nesneleri tek bir işlemde tespit etmeyi 
amaçlayan bir nesne tanıma algoritmasıdır. Diğer yöntemlerde olduğu gibi birden fazla 
aşamalı işlem yerine YOLO, tüm resmi bir kerede işler. Bu sayede gerçek zamanlı sonuçlar 
verebilir. 
YOLO, resmi belirli sayıda grid (ızgara) hücresine böler. Her hücre, içindeki nesneye ait 
olup olmadığını kontrol eder ve varsa, nesnenin koordinatlarını ve sınıfını tahmin eder. 
● YOLOv3: 
○ 3 farklı ölçekte tahmin (13x13, 26x26, 52x52). 
○ Küçük nesne tespitinde daha başarılı. 
○ Darknet-53 ile derin özellik çıkarımı. 
● YOLOv8: 
○ Tamamen anchor-free. 
○ Kolay kullanım: model.predict(), model.train() gibi. 
○ Segmentasyon ve sınıflandırma için de kullanılabilir. 
○ Ultralytics resmi destek veriyor. 
�
�
 YOLO’nun Temel Adımları: 
1. Girdiyi sabit boyutlu (örneğin 416x416) olarak alır. 
2. Görüntüyü S x S'lik bir grid'e böler. 
3. Her grid hücresi, kendisine düşen alan içinde nesne olup olmadığını tahmin eder. 
4. Her hücre: 
○ N adet bounding box (kutu) üretir. 
○ Her kutu için: koordinatlar (x, y, w, h) + nesne olasılığı (confidence). 
○ Ayrıca, kutunun hangi sınıfa ait olduğunu (class probabilities) tahmin eder.  
Grid Sistemi 
Girdi görüntüsü S x S boyutunda bir grid'e bölünür. Örneğin 13x13 veya 7x7. 
Her hücre, görüntünün sadece o bölgesinden sorumludur. Eğer nesnenin merkezi o grid 
hücresine denk geliyorsa, hücre o nesneden sorumlu olur. 
Her hücre şunları tahmin eder: 
● Bounding box (konum bilgileri): x, y, w, h 
● Confidence score: O kutunun gerçekten bir nesne içerip içermediğini belirten olasılık 
● Sınıf tahmini: Kutunun hangi nesneye ait olduğu (örneğin: insan, araba, köpek vs.) 
○  
�
�
 Anchor Boxes 
● Farklı şekil ve boyutlardaki nesneleri daha iyi temsil etmek için sabit kutu 
şablonlarıdır. 
● Model bu sabit kutular üzerinden sapma (offset) öğrenir. 
● Örneğin: YOLOv3 genelde 9 anchor box kullanır (3 ölçek, her ölçekte 3 kutu). 
YOLOv8 ile Anchor-Free Mimari 
YOLOv8'de anchor-box kullanılmaz. Bunun yerine doğrudan koordinat tahmini yapılır. Bu 
sayede: 
● Eğitim süreci hızlanır. 
● Daha az ayarlama gerektirir. 
● Küçük veri setlerinde daha iyi performans gösterir. 
�
�
 Bu Hatalı Tahminin Nedenleri: 
1. Modelin Eğitim Verisi Sınırlı 
● YOLOv8’in varsayılan modeli (yolov8n.pt) COCO dataset ile eğitilmiştir. 
● COCO veri setinde “flower” (çiçek) sınıfı yoktur, ama “umbrella” vardır. 
● Model benzer şekli en yakın tanıdığı sınıfa (umbrella) benzetmiş olabilir. 
2. Benzer Görsel Özellikler 
● Çiçeğin yayvan ve yuvarlak yaprakları uzaktan bakıldığında bir şemsiyeye benziyor 
olabilir. 
● YOLO, renk, şekil ve desenlere bakarak tahmin yapar, ama insan gibi anlamaz. 
3. Düşük Güven Skoru : %63 doğruluk değeri düşük bir değerdir. Modelin çok emin olmadığını
