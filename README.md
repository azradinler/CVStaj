# CVStaj
 
Classification:
- Görüntülere veya bölgelere etiketler atar
- Görüntü içeriğinin kapsamlı bir şekilde anlaşılmasını sağlar
- Görüntü etiketleme ve etiketlemeyi etkinleştirir

Object Detection:
- Belirli nesneleri ve konumlarını tanımlar 
- Hassas nesne yerelleştirmesi için sınırlayıcı kutular kullanır 
- Video gözetimi ve güvenlik izlemeyi etkinleştirir 
- Tarımda ürün izleme ve zararlı tespiti için yardımcı olur 

Segmentation:
- Nesne sınırları ve bölgeleri hakkında ayrıntılı bilgi sağlar
                                                       
Yolo 
Her bir ızgara kendi içinde, alanda nesnenin olup olmadığını, varsa orta noktasının içinde olup olmadığını, orta noktası da içindeyse uzunluğunu, yüksekliğini ve hangi sınıftan olduğunu bulmakla sorumlu. 
 
Buna göre YOLO her ızgara için ayrı bir tahmin vektörü oluşturur. Bunların her birinin içinde:
Güven skoru: Bu skor modelin geçerli ızgara içinde nesne bulunup bulunmadığından ne kadar emin olduğunu gösterir. (0 ise kesinlikle yok 1 ise kesinlikle var) Eğer nesne olduğunu düşünürse de bu nesnenin gerçekten o nesne olup olmadığından ve etrafındaki kutunun koordinatlarından ne kadar emin olduğunu gösterir.
Bx: Nesnenin orta noktasının x koordinatı
By: Nesnenin orta noktasının y koordinatı
Bw: Nesnenin genişliği
Bh: Nesnenin yüksekliği
Bağlı Sınıf Olasılığı: Modelimizde kaç farklı sınıf varsa o kadar sayıda tahmin değeri. 
Güven skoru = Kutu Güven Skoru x Bağlı Sınıf Olasılığı
Kutu Güven Skoru = P(nesne) . IoU
P(nesne) = Kutunun nesneyi kapsayıp kapsamadığının olasılığı. (Yani nesne var mı yok mu?)
IoU = Ground truth ile tahmin edilmiş kutu arasındaki IoU değeri

YOLO Algoritması Açıklaması
YOLO (You Only Look Once) algoritması, giriş görüntüsünü SxS boyutunda grid (ızgara) hücrelerine böler. Her bir grid hücresi, potansiyel olarak bir veya birden fazla bounding box (sınırlayıcı kutu) ve bu kutuların güvenirlik skorlarını (confidence) tahmin eder. Aynı zamanda her grid hücresi için sınıf olasılıkları (class probability map) da oluşturulur. Bu yapı, üst üste binen nesnelerin daha doğru tespiti için önemlidir.
Her grid hücresi:
•	Bir veya birden fazla bounding box (x, y koordinatları, genişlik ve yükseklik dahil),
•	Bu kutulara ait confidence skorları,
•	Ve sınıf olasılıklarını üretir.
YOLO algoritması, bu görevleri CNN (Convolutional Neural Network) tabanlı bir mimari ile gerçekleştirir.
 
Anchor box, YOLO gibi nesne tespit algoritmalarında, önceden tanımlanmış kutu şablonlarıdır. Bu kutular, görüntüdeki farklı nesne boyutlarını ve oranlarını daha iyi tespit edebilmek için kullanılır.
 Her grid hücresine birden fazla anchor box atanır (örneğin 3 adet).
Bu anchor box'lar, önceden analiz edilmiş veri setine göre farklı en-boy oranlarına (aspect ratio) sahiptir.
Model, her anchor box üzerinden:
•	Nesne olup olmadığını (objectness score),
•	Sınıfını (class probability),
•	Anchor box'a göre pozisyon düzeltmesini (x, y, w, h) tahmin eder.
Eğitim sırasında, nesneye en yakın anchor box seçilir (IoU en yüksek olan) ve geri kalanı göz ardı edilir.
Kutu Koordinatları ve Tespit Süreci
•	Bounding box koordinatları, her kutunun merkez noktası baz alınarak hesaplanır. Kutu genişliği ve yüksekliği bu merkeze göre belirlenir.
•	Nesne tespiti için yalnızca merkez noktası bir grid hücresine düşen kutular dikkate alınır. Diğer hücreler için tahminler sıfır (0) olarak kabul edilir.
•	Son olarak, tespit edilen kutular üzerinde IoU (Intersection over Union) skoru hesaplanarak, önceden belirlenmiş bir eşik değer (threshold) ile karşılaştırma yapılır. Böylece en güvenilir kutular seçilir ve tespit sonuçları oluşturulur.

Versiyonlar Arası Farklar
YOLOv1: Her grid hücresi yalnızca tek nesne için tahmin yapabilir. Eğer bir hücrede birden fazla nesne varsa bu durum performans kaybına neden olur.
YOLOv3 ve sonrası: Multilabel sınıflandırmayı destekler. Örneğin bir nesne hem "animal" hem de "dog" olarak etiketlenebilir. Bu, daha karmaşık sınıflandırma senaryolarında avantaj sağlar.anchor based. Nesne kutu tespiti Bounding box ,IoU tababnlı filtreleme.
YOLOv8:Anchor free .Nesne kutu tespiti IoU +NMS + Confidence threshold 

YOLO  ile insan tespiti 
YOLO, bir görüntüyü tek seferde (tek bakışta) analiz ederek içerisinde bulunan nesneleri ve konumlarını tahmin eder. Özellikle "person" sınıfı, yaygın veri setleri (örneğin COCO) içinde önemli bir sınıftır ve bu sayede insan tespiti için oldukça etkilidir.
Sınıf olasılıklarını tahmin eder (örneğin: insan, araba, kedi vb.).
Bounding box (x, y, w, h) değerlerini ve güven skorunu tahmin eder.
Filtreleme Uygular:
Belirli bir eşik değerinden düşük skorlar elenir.
Aynı kişiye ait birden fazla kutuyu engellemek için NMS (Non-Maximum Suppression) uygulanır.
Sonuç: Tespit edilen insanlara ait kutular ve sınıf etiketleri çıktı olarak verilir.


