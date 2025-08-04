# CVStaj
##GÖRÜNTÜ İŞLEME - OPENCV

CNN Nedir?

Evrişimsel Sinir Ağları (Convolutional Neural Networks - CNN), özellikle
görüntü işleme ve bilgisayar görüşü alanlarında devrim yaratan derin
öğrenme modelleridir. Geleneksel yapay sinir ağlarının aksine, CNN’ler
görüntülerin iki boyutlu (veya üç boyutlu) yapısını doğrudan
işleyebilecek şekilde tasarlanmıştır. Bu özellikleri sayesinde,
görüntülerdeki hiyerarşik örüntüleri, kenarları, köşeleri, dokuları ve
daha karmaşık şekilleri otomatik olarak öğrenme yeteneğine sahiptirler.

Bir CNN’in temel amacı, bir görüntüyü girdi olarak alıp, o görüntüdeki
nesneleri, yüzleri veya diğer görsel unsurları tanımlamak,
sınıflandırmak veya tespit etmektir. Bu süreç, insan beyninin görsel
korteksinin çalışma prensibinden esinlenerek geliştirilmiştir. Görsel
korteks, görüntüyü farklı katmanlarda işleyerek basit özelliklerden
karmaşık özelliklere doğru bir hiyerarşi oluşturur. CNN’ler de benzer
bir katmanlı yapıya sahiptir.

CNN’in Temel Bileşenleri
------------------------

Bir CNN genellikle üç ana tür katmandan oluşur:

1.  **Evrişim Katmanları (Convolutional Layers):** Görüntüdeki özellik
    haritalarını (feature maps) çıkarmak için kullanılır.
2.  **Havuzlama Katmanları (Pooling Layers):** Özellik haritalarının
    boyutunu küçültmek ve önemli bilgileri korumak için kullanılır.
3.  **Tam Bağlantılı Katmanlar (Fully Connected Layers):** Öğrenilen
    özellikleri kullanarak sınıflandırma veya regresyon gibi nihai
    tahminleri yapmak için kullanılır.

Bu katmanlar arasına genellikle **Aktivasyon Fonksiyonları (Activation
Functions)** eklenir. Aktivasyon fonksiyonları, ağa doğrusal olmayan bir
yetenek kazandırarak daha karmaşık ilişkileri öğrenmesini sağlar.

CNN’ler, görüntü tanıma, nesne tespiti, yüz tanıma, tıbbi görüntü
analizi ve otonom sürüş gibi birçok alanda başarıyla uygulanmaktadır. Bu
notebook’ta, CNN’in bu temel katmanlarını ve çalışma prensibini detaylı
bir şekilde inceleyeceğiz.

Evrişim Katmanı (Convolutional Layer)
-------------------------------------

Evrişim katmanı, bir CNN’in kalbidir ve görüntülerden özellik çıkarmak
için kullanılır. Bu katman, bir girdi görüntüsü üzerinde küçük bir
filtre (kernel) veya çekirdek kullanarak evrişim (convolution) işlemi
gerçekleştirir. Filtre, görüntünün farklı bölgeleri üzerinde
kaydırılarak (stride) yerel özellikler (kenarlar, köşeler, dokular gibi)
tespit eder.

### Evrişim İşleminin Matematiksel Anlatımı

Evrişim işlemi, iki fonksiyonun (bu durumda girdi görüntüsü ve filtre)
birbiri üzerindeki kaydırılmış çarpımlarının toplamı olarak tanımlanır.
İki boyutlu bir görüntü için evrişim işlemi aşağıdaki gibi ifade edilir:

$$ (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n) $$

Burada: - $I$: Girdi görüntüsü - $K$: Filtre (kernel) - $(i, j)$: Çıktı
özellik haritasındaki pikselin koordinatları - $m, n$: Filtrenin
boyutları üzerindeki indeksler

Bu formül, filtrenin her bir konumda girdi görüntüsü üzerindeki ilgili
piksellerle eleman bazında çarpılıp toplanmasıyla bir çıktı değeri
üretildiğini gösterir. Bu çıktı değerleri, **özellik haritası (feature
map)** veya **aktivasyon haritası (activation map)** olarak
adlandırılır.

### Evrişim Katmanının Çalışma Prensibi

1.  **Filtreler (Kernels):** Her bir filtre, görüntünün belirli bir
    özelliğini (örneğin, yatay kenarlar, dikey kenarlar) tespit etmek
    üzere tasarlanmış küçük bir matristir. CNN eğitimi sırasında bu
    filtrelerin değerleri otomatik olarak öğrenilir.
2.  **Kaydırma (Stride):** Filtrenin girdi görüntüsü üzerinde ne kadar
    adımda kaydırılacağını belirler. Daha büyük bir kaydırma değeri,
    çıktı özellik haritasının boyutunu küçültür.
3.  **Doldurma (Padding):** Görüntünün kenarlarına sıfır pikseller
    eklenerek çıktı özellik haritasının boyutunun korunması veya belirli
    bir boyuta getirilmesi sağlanır. Bu, özellikle kenarlardaki
    bilgilerin kaybolmasını önler.
4.  **Çoklu Filtreler:** Bir evrişim katmanı genellikle birden fazla
    filtre kullanır. Her bir filtre farklı bir özelliği öğrenir ve
    farklı bir özellik haritası üretir. Bu özellik haritaları bir araya
    getirilerek katmanın nihai çıktısını oluşturur.

Örneğin, 5x5 boyutunda bir girdi görüntüsü ve 3x3 boyutunda bir filtre
ile kaydırma (stride) 1 ve doldurma (padding) 0 olan bir evrişim işlemi
aşağıdaki gibi görselleştirilebilir:

![Evrişim
İşlemi](attachment:/home/ubuntu/upload/search_images/KOrA5IzJsODU.png)
*Görsel 1: Evrişim İşlemi \[1\]*

Bu işlem sonucunda, girdi görüntüsündeki belirli bir özelliğin varlığı
veya yokluğu, çıktı özellik haritasındaki yüksek veya düşük değerlerle
temsil edilir.

Havuzlama Katmanı (Pooling Layer)
---------------------------------

Havuzlama katmanı, evrişim katmanlarından sonra genellikle kullanılan ve
özellik haritalarının boyutunu küçültmek (downsampling) için tasarlanmış
bir katmandır. Bu boyut küçültme işlemi, modelin hesaplama yükünü
azaltır, aşırı öğrenmeyi (overfitting) engellemeye yardımcı olur ve
modelin konum değişikliklerine karşı daha dirençli olmasını sağlar
(translation invariance).

### Havuzlama İşleminin Matematiksel Anlatımı

Havuzlama işlemi, bir pencere (genellikle 2x2 veya 3x3) kullanarak
özellik haritasının belirli bir bölgesindeki değerleri tek bir temsili
değerle değiştirir. En yaygın havuzlama türleri şunlardır:

1.  **Maksimum Havuzlama (Max Pooling):** Pencere içindeki en büyük
    değeri seçer.
    $$ O_{i,j} = \max_{m,n \in Window} I_{i \cdot S + m, j \cdot S + n} $$
    Burada:
    -   $O_{i,j}$: Çıktı havuzlanmış haritadaki değer
    -   $I$: Girdi özellik haritası
    -   $S$: Kaydırma (stride) boyutu
    -   $Window$: Havuzlama penceresi
2.  **Ortalama Havuzlama (Average Pooling):** Pencere içindeki tüm
    değerlerin ortalamasını alır.
    $$ O_{i,j} = \frac{1}{|Window|} \sum_{m,n \in Window} I_{i \cdot S + m, j \cdot S + n} $$
    Burada:
    -   $|Window|$: Pencere içindeki eleman sayısı

### Havuzlama Katmanının Çalışma Prensibi

Havuzlama katmanı, evrişim katmanında olduğu gibi filtreler kullanmaz,
bunun yerine belirli bir pencere boyutu ve kaydırma (stride) ile
çalışır. Pencere, özellik haritası üzerinde kaydırılır ve her pencere
konumunda belirlenen havuzlama işlemi (maksimum veya ortalama)
uygulanır.

Örneğin, 4x4 boyutunda bir özellik haritası üzerinde 2x2 boyutunda bir
pencere ve 2 kaydırma (stride) ile maksimum havuzlama işlemi aşağıdaki
gibi görselleştirilebilir:

![Maksimum
Havuzlama](attachment:/home/ubuntu/upload/search_images/KOrA5IzJsODU.png)
*Görsel 2: Maksimum Havuzlama İşlemi \[1\]*

Bu işlem sonucunda, özellik haritasının boyutu küçülürken, en önemli
özellik bilgileri (maksimum havuzlama durumunda en belirgin özellikler)
korunmuş olur. Ortalama havuzlama ise daha yumuşak bir boyut küçültme
sağlar ve tüm bilgilerin ortalamasını alarak daha genel bir temsil
oluşturur.

Aktivasyon Katmanı (Activation Layer)
-------------------------------------

Aktivasyon katmanları, bir sinir ağının doğrusal olmayan ilişkileri
öğrenmesini sağlayan kritik bileşenlerdir. Evrişim ve havuzlama
katmanları genellikle doğrusal işlemler gerçekleştirir. Ancak gerçek
dünyadaki veriler ve özellikler genellikle doğrusal olmayan bir yapıya
sahiptir. Aktivasyon fonksiyonları, ağın bu doğrusal olmayan
karmaşıklıkları modellemesine olanak tanır.

### Aktivasyon Fonksiyonlarının Önemi

Eğer bir sinir ağında aktivasyon fonksiyonları kullanılmazsa, ağ ne
kadar derin olursa olsun (ne kadar çok katmanı olursa olsun), sadece
doğrusal bir dönüşüm gerçekleştirecektir. Bu durum, ağın karmaşık
örüntüleri öğrenme yeteneğini ciddi şekilde sınırlar. Aktivasyon
fonksiyonları, her bir nöronun çıktısına doğrusal olmayan bir dönüşüm
uygulayarak ağın daha karmaşık fonksiyonları temsil etmesini sağlar.

### Yaygın Aktivasyon Fonksiyonları

1.  **ReLU (Rectified Linear Unit):** En yaygın kullanılan aktivasyon
    fonksiyonlarından biridir. Basitliği ve hesaplama verimliliği
    nedeniyle tercih edilir. $$ f(x) = \max(0, x) $$

    -   Eğer girdi pozitifse, çıktıyı olduğu gibi verir.
    -   Eğer girdi negatifse, çıktıyı sıfır yapar.

2.  **Sigmoid:** Çıktıyı 0 ile 1 arasına sıkıştırır. Özellikle ikili
    sınıflandırma problemlerinde çıktı katmanında kullanılır.
    $$ f(x) = \frac{1}{1 + e^{-x}} $$

3.  **Tanh (Hyperbolic Tangent):** Çıktıyı -1 ile 1 arasına sıkıştırır.
    Sigmoid fonksiyonuna benzer, ancak sıfır merkezli olması nedeniyle
    bazı durumlarda daha iyi performans gösterebilir.
    $$ f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $$

### Aktivasyon Katmanının Çalışma Prensibi

Aktivasyon fonksiyonu, evrişim katmanının çıktısı olan özellik
haritasındaki her bir piksel değerine bağımsız olarak uygulanır. Bu
işlem, özellik haritasındaki değerlere doğrusal olmayan bir dönüşüm
uygulayarak ağın daha karmaşık özellikleri öğrenmesine yardımcı olur.

Tam Bağlantılı Katman (Fully Connected Layer)
---------------------------------------------

Tam bağlantılı katmanlar (Dense katmanlar olarak da bilinir), geleneksel
yapay sinir ağlarındaki katmanlara benzer. Bir CNN mimarisinde, evrişim
ve havuzlama katmanlarından sonra gelirler. Bu katmanların temel amacı,
evrişim katmanları tarafından çıkarılan yüksek seviyeli özellikleri
kullanarak sınıflandırma veya regresyon gibi nihai tahminleri yapmaktır.

### Tam Bağlantılı Katmanın Çalışma Prensibi

1.  **Düzleştirme (Flattening):** Evrişim ve havuzlama katmanlarından
    gelen iki boyutlu (veya üç boyutlu) özellik haritaları, tam
    bağlantılı katmanlara beslenmeden önce tek boyutlu bir vektöre
    dönüştürülür. Bu işleme düzleştirme denir.
2.  **Ağırlıklar ve Sapmalar:** Düzleştirilmiş vektördeki her bir nöron,
    tam bağlantılı katmandaki her bir nörona bir ağırlık ve sapma (bias)
    ile bağlanır. Bu, her bir girdi özelliğinin çıktı üzerindeki
    etkisini öğrenmesini sağlar.
3.  **Çıktı:** Tam bağlantılı katmanların çıktısı, genellikle bir
    sınıflandırma problemi için sınıf olasılıkları (softmax aktivasyon
    fonksiyonu ile) veya bir regresyon problemi için sürekli bir değer
    (doğrusal aktivasyon fonksiyonu ile) olabilir.

Tam bağlantılı katmanlar, ağın öğrenilen özellikleri bir araya getirerek
karmaşık karar sınırları oluşturmasına ve nihai tahmini yapmasına olanak
tanır. Örneğin, bir görüntüdeki kediyi veya köpeği sınıflandırmak için,
evrişim katmanları kedinin veya köpeğin özelliklerini çıkarırken, tam
bağlantılı katmanlar bu özellikleri kullanarak görüntünün hangi sınıfa
ait olduğuna karar verir.

![CNN
Mimarisi](attachment:/home/ubuntu/upload/search_images/a0d6sF3QG0Ro.png)
*Görsel 3: Tipik Bir CNN Mimarisi \[2\]*

Bu görselde, evrişim, havuzlama ve tam bağlantılı katmanların bir CNN
içerisindeki genel akışı gösterilmektedir. Görüntü, evrişim ve havuzlama
katmanlarından geçerek özellik çıkarımı yapılır, ardından
düzleştirilerek tam bağlantılı katmanlara beslenir ve son olarak çıktı
katmanında sınıflandırma yapılır.

Detection, Classification, Segmentation Nedir?
==============================================

Bilgisayar görüşü alanında, görüntülerden anlamlı bilgi çıkarmak için
çeşitli görevler bulunur. Bu görevler arasında sınıflandırma
(Classification), nesne tespiti (Object Detection) ve segmentasyon
(Segmentation) en temel ve yaygın olanlarıdır. Her bir görev, farklı bir
detay seviyesinde bilgi sağlar ve farklı uygulama senaryolarına hizmet
eder.

Görüntü Sınıflandırma (Image Classification)
--------------------------------------------

Görüntü sınıflandırma, bir görüntüdeki ana nesnenin veya sahnenin ne
olduğunu belirleme görevidir. Bu görevde, girdi olarak bir görüntü
alınır ve çıktı olarak bu görüntünün ait olduğu sınıfın etiketi (label)
verilir. Yani, görüntüdeki her bir pikselin değil, görüntünün bir bütün
olarak hangi kategoriye girdiğine karar verilir.

### Çıktı Türü ve Uygulama Senaryoları

-   **Çıktı Türü:** Tek bir sınıf etiketi (örneğin,

‘kedi’, ‘köpek’, ‘araba’). - **Uygulama Senaryoları:** - **Spam E-posta
Filtreleme:** Gelen e-postaların içeriğine göre spam olup olmadığını
belirleme. - **Tıbbi Teşhis:** Röntgen veya MR görüntülerinin belirli
bir hastalığı (örneğin, tümör) içerip içermediğini sınıflandırma. -
**İçerik Moderasyonu:** Resimlerin veya videoların uygunsuz içerik
barındırıp barındırmadığını otomatik olarak belirleme. - **Ürün
Kategorizasyonu:** E-ticaret sitelerinde ürün görsellerini doğru
kategoriye atama.

![Görüntü Sınıflandırma
Örneği](attachment:/home/ubuntu/upload/search_images/d74M73CbvvFc.png)
*Görsel 4: Görüntü Sınıflandırma, Nesne Tespiti ve Anlamsal Segmentasyon
Karşılaştırması \[3\]*

Bu görselde, bir görüntünün sadece ‘kedi’ olarak sınıflandırılması,
görüntü sınıflandırmanın temelini oluşturur. Görüntüdeki nesnenin tam
konumunu veya birden fazla nesneyi ayırt etme yeteneği yoktur.

Nesne Tespiti (Object Detection)
--------------------------------

Nesne tespiti, bir görüntüdeki belirli nesnelerin hem konumunu
(sınırlayıcı kutular - bounding boxes ile) hem de sınıfını belirleme
görevidir. Görüntü sınıflandırmadan farklı olarak, nesne tespiti bir
görüntüde birden fazla nesne olduğunda her bir nesneyi ayrı ayrı
tanımlayabilir ve konumlandırabilir.

### Çıktı Türü ve Uygulama Senaryoları

-   **Çıktı Türü:** Her tespit edilen nesne için bir sınırlayıcı kutu
    (x, y koordinatları, genişlik, yükseklik) ve bir sınıf etiketi
    (örneğin, ‘kedi’, ‘köpek’, ‘araba’).
-   **Uygulama Senaryoları:**
    -   **Otonom Araçlar:** Yoldaki diğer araçları, yayaları, trafik
        işaretlerini ve şeritleri tespit etme.
    -   **Güvenlik ve Gözetim:** Kalabalık alanlarda şüpheli nesneleri
        veya kişileri tespit etme.
    -   **Perakende:** Mağazalardaki raf düzenini izleme, ürün
        stoklarını takip etme ve müşteri davranışlarını analiz etme.
    -   **Yüz Tanıma Sistemleri:** Bir görüntüdeki yüzleri tespit etme
        ve ardından bu yüzleri tanıma.

![Nesne Tespiti
Örneği](attachment:/home/ubuntu/upload/search_images/nhp9ZXCWLJ2z.png)
*Görsel 5: Görüntü Sınıflandırma, Nesne Tespiti ve Anahtar Nokta Tespiti
Karşılaştırması \[4\]*

Bu görselde, bir görüntüdeki birden fazla kedinin ve köpeğin ayrı ayrı
sınırlayıcı kutularla tespit edildiği ve etiketlendiği görülmektedir.
Bu, nesne tespitinin sınıflandırmadan daha detaylı bilgi sağladığını
gösterir.

Anlamsal Segmentasyon (Semantic Segmentation)
---------------------------------------------

Anlamsal segmentasyon, bir görüntüdeki her bir pikseli belirli bir
sınıfa atama görevidir. Bu, nesne tespitinden daha granüler bir
yaklaşımdır; çünkü sadece nesnenin konumunu değil, aynı zamanda nesnenin
tam şeklini ve sınırlarını da belirler. Her pikselin ait olduğu sınıf
belirlenir, ancak aynı sınıfa ait farklı nesneler arasında ayrım
yapılmaz (örneğin, iki farklı kedi aynı ‘kedi’ sınıfına ait pikseller
olarak etiketlenir).

### Çıktı Türü ve Uygulama Senaryoları

-   **Çıktı Türü:** Görüntüyle aynı boyutta bir maske veya harita; her
    pikselin ait olduğu sınıfı temsil eden bir etiket (örneğin,
    ‘gökyüzü’, ‘yol’, ‘ağaç’, ‘insan’).
-   **Uygulama Senaryoları:**
    -   **Tıbbi Görüntüleme:** Tümörlerin, organların veya diğer
        biyolojik yapıların kesin sınırlarını belirleme.
    -   **Otonom Sürüş:** Yol, kaldırım, araçlar ve yayalar gibi farklı
        bölgeleri piksel düzeyinde ayırma.
    -   **Uydu Görüntüleri Analizi:** Arazi kullanımını (orman, su,
        şehir) veya ekin alanlarını haritalama.
    -   **Artırılmış Gerçeklik (AR):** Gerçek dünya nesnelerinin üzerine
        sanal nesneleri doğru bir şekilde yerleştirmek için sahneyi
        anlama.

![Anlamsal Segmentasyon
Örneği](attachment:/home/ubuntu/upload/search_images/dToLWy551CAj.jpg)
*Görsel 6: Anlamsal Segmentasyon ve Nesne Tespiti Karşılaştırması \[5\]*

Bu görselde, kedinin ve köpeğin sadece sınırlayıcı kutularla değil, aynı
zamanda piksel düzeyinde kesin sınırlarla ayrıldığı görülmektedir. Bu,
anlamsal segmentasyonun en detaylı çıktı türünü sağladığını gösterir.

Özet Karşılaştırma
------------------

| Görev                     | Çıktı Türü                              | Detay Seviyesi         | Örnek Uygulama                                                              |
|:--------------------------|:----------------------------------------|:-----------------------|:----------------------------------------------------------------------------|
| **Görüntü Sınıflandırma** | Tek bir sınıf etiketi                   | Düşük (Görüntü Geneli) | Bir görüntünün ‘kedi’ mi ‘köpek’ mi olduğunu belirleme                      |
| **Nesne Tespiti**         | Sınırlayıcı kutular ve sınıf etiketleri | Orta (Nesne Konumu)    | Görüntüdeki tüm kedileri ve köpekleri bulma ve konumlandırma                |
| **Anlamsal Segmentasyon** | Her piksel için sınıf etiketi (maske)   | Yüksek (Piksel Düzeyi) | Görüntüdeki her bir kedinin ve köpeğin tam şeklini ve sınırlarını belirleme |

Bu üç görev, bilgisayar görüşü alanındaki farklı ihtiyaçlara cevap verir
ve genellikle birbirini tamamlayıcı niteliktedir. Birçok karmaşık
bilgisayar görüşü sistemi, bu görevlerin bir kombinasyonunu kullanarak
daha kapsamlı analizler yapar.

YOLO Teorisi
============

YOLO (You Only Look Once), gerçek zamanlı nesne tespiti için
geliştirilmiş devrim niteliğinde bir derin öğrenme modelidir. Geleneksel
nesne tespit yöntemlerinin (örneğin, R-CNN, Fast R-CNN) aksine, YOLO,
nesne tespiti problemini tek bir evrişimsel ağ üzerinden çözer. Bu, onu
çok daha hızlı ve gerçek zamanlı uygulamalar için ideal hale getirir.

YOLO Mimarisi ve Temel Yapısı
-----------------------------

YOLO, bir görüntüyü girdi olarak alır ve doğrudan sınırlayıcı kutu
koordinatlarını ve sınıf olasılıklarını tahmin eder. Bu, iki aşamalı
(region proposal + classification/regression) sistemlerin aksine, tek
aşamalı bir yaklaşımdır. Temel YOLO mimarisi genellikle aşağıdaki
bileşenleri içerir:

1.  **Evrişimsel Katmanlar:** Görüntüden özellik çıkarmak için
    kullanılır.
2.  **Tam Bağlantılı Katmanlar:** Çıkarılan özelliklerden sınırlayıcı
    kutu koordinatlarını ve sınıf olasılıklarını tahmin etmek için
    kullanılır.

YOLO, girdi görüntüsünü bir ızgaraya (grid) böler. Her bir ızgara
hücresi, eğer bir nesnenin merkezi o hücreye düşüyorsa, o nesneyi tespit
etmekten sorumlu olur. Her ızgara hücresi, belirli sayıda sınırlayıcı
kutu (bounding box) ve bu kutular için sınıf olasılıkları tahmin eder.

### Izgara (Grid) Mantığı

YOLO, girdi görüntüsünü $S \times S$ boyutunda bir ızgaraya böler.
Örneğin, $7 \times 7$ bir ızgara kullanılıyorsa, görüntü 49 eşit hücreye
ayrılır. Eğer bir nesnenin merkezi bu hücrelerden birine düşerse, o
hücre o nesnenin tespitinden sorumlu olur. Her bir ızgara hücresi için
şunlar tahmin edilir:

-   **Sınırlayıcı Kutu Koordinatları:** Her bir sınırlayıcı kutu için
    $(x, y, w, h)$ koordinatları tahmin edilir. Burada $(x, y)$ kutunun
    merkezini, $w$ genişliğini ve $h$ yüksekliğini temsil eder.
-   **Güven Skoru (Confidence Score):** Bu skor, kutunun bir nesne
    içerip içermediğini ve içeriyorsa ne kadar doğru tahmin edildiğini
    gösterir. $P(Object) \times IOU_{pred}^{truth}$ olarak hesaplanır.
-   **Sınıf Olasılıkları:** Her bir sınırlayıcı kutu için, o kutunun
    belirli bir sınıfa ait olma olasılıkları tahmin edilir.
    $P(Class_i | Object)$.

### Anchor Box (Çapa Kutusu) Mantığı

YOLOv2 ve sonraki versiyonlarda tanıtılan Anchor Box kavramı, modelin
farklı boyut ve oranlardaki nesneleri daha iyi tespit etmesini sağlar.
Eğitimden önce, veri kümesindeki nesnelerin boyut ve oran dağılımına
göre belirli sayıda (örneğin, 5 veya 9) önceden tanımlanmış çapa
kutuları belirlenir. Model, her bir ızgara hücresi için bu çapa
kutularını referans alarak nesnelerin sınırlayıcı kutularını tahmin
eder. Yani, her bir tahmin, bir çapa kutusunun ofsetleri (kaydırmaları)
ve ölçeklendirmeleri olarak yorumlanır.

Bu sayede, model aynı ızgara hücresinden birden fazla nesneyi tespit
edebilir ve farklı şekillerdeki nesneleri daha doğru bir şekilde
kapsayabilir.

![YOLO
Mimarisi](attachment:/home/ubuntu/upload/search_images/4E616Ch4WPDp.png)
*Görsel 7: YOLO Mimarisi Genel Bakış \[6\]*

Bu görsel, YOLO’nun tek bir ağ üzerinden nasıl nesne tespiti yaptığını
ve ızgara sistemini genel hatlarıyla göstermektedir.

YOLO Versiyonları Arası Farklar (v3 – v8)
-----------------------------------------

YOLO, ilk tanıtıldığı günden bu yana sürekli olarak geliştirilmiş ve
birçok farklı versiyonu piyasaya sürülmüştür. Her yeni versiyon,
genellikle hız ve doğruluk arasında daha iyi bir denge sağlamayı
hedeflerken, mimaride ve eğitim stratejilerinde önemli iyileştirmeler
getirmiştir.

### YOLOv3

-   **Mimari İyileştirmeler:** Darknet-53 adı verilen daha derin bir
    omurga ağı (backbone network) kullanır. Bu, daha fazla evrişimsel
    katman ve daha zengin özellik çıkarımı anlamına gelir.
-   **Çoklu Ölçek Tespiti:** Farklı boyutlardaki nesneleri tespit etmek
    için üç farklı ölçekte tahmin yapar. Bu, küçük nesnelerin tespitinde
    önemli bir iyileşme sağlar.
-   **Anchor Box İyileştirmeleri:** Her bir ölçek için farklı boyutlarda
    çapa kutuları kullanır.
-   **Sınıflandırma:** Softmax yerine lojistik regresyon kullanarak
    çoklu etiket sınıflandırmasına olanak tanır.

### YOLOv4

-   **Mimari İyileştirmeler:** CSPDarknet53 omurga ağı, SPP (Spatial
    Pyramid Pooling) ve PANet (Path Aggregation Network) gibi yeni
    bileşenler eklenmiştir. Bu, özellik çıkarımını ve özelliklerin
    birleşimini iyileştirir.
-   **Veri Artırma Teknikleri:** Mosaic veri artırma gibi yeni teknikler
    kullanılarak modelin genelleme yeteneği artırılmıştır.
-   **Optimizasyon Teknikleri:** Mish aktivasyon fonksiyonu, DropBlock
    regularizasyonu gibi optimizasyon teknikleri ile eğitim süreci
    iyileştirilmiştir.

### YOLOv5

-   **Geliştirici:** Ultralytics tarafından geliştirilmiştir. PyTorch
    tabanlıdır.
-   **Model Boyutları:** Farklı performans ve hız ihtiyaçlarına göre
    çeşitli model boyutları (N, S, M, L, X) sunar.
-   **Kullanım Kolaylığı:** Daha modüler bir yapıya sahiptir ve
    kullanımı daha kolaydır. Eğitim ve dağıtım süreçleri
    basitleştirilmiştir.
-   **Performans:** Genellikle daha küçük boyutlarda bile yüksek
    doğruluk ve hız sunar.

### YOLOv6

-   **Geliştirici:** Meituan tarafından geliştirilmiştir.
-   **Donanım Dostu:** Özellikle donanım hızlandırması için optimize
    edilmiştir, bu da daha hızlı çıkarım süreleri sağlar.
-   **Mimari:** Yeniden tasarlanmış bir omurga ve boyun (neck) yapısı
    kullanır.

### YOLOv7

-   **Geliştirici:** YOLOv4’ün orijinal yazarları tarafından
    geliştirilmiştir.
-   **Hız ve Doğruluk:** Önceki versiyonlara göre daha iyi hız-doğruluk
    dengesi sunar.
-   **Genişletilmiş Bileşenler:** Genişletilmiş ve bileşik ölçeklendirme
    yöntemleri ile modelin kapasitesi artırılmıştır.

### YOLOv8

-   **Geliştirici:** Ultralytics tarafından geliştirilmiştir.
-   **Yeni Mimari:** Önceki YOLO versiyonlarından farklı olarak, tamamen
    yeni bir omurga ve boyun yapısı kullanır. Anchor-free (çapa kutusuz)
    bir yaklaşım benimser, bu da modelin tasarımını basitleştirir ve
    daha esnek hale getirir.
-   **Çoklu Görev Yeteneği:** Nesne tespiti, segmentasyon ve poz tahmini
    gibi birden fazla görevi destekler.
-   **Performans:** Genellikle daha yüksek doğruluk ve daha hızlı
    çıkarım süreleri sunar.

![YOLO Versiyonları
Karşılaştırması](attachment:/home/ubuntu/upload/search_images/bEUng8a8sGkw.webp)
*Görsel 8: YOLO Versiyonları Arası Performans Karşılaştırması \[7\]*

Bu görsel, farklı YOLO versiyonlarının performansını (doğruluk ve hız)
karşılaştırmalı olarak göstermektedir. Her yeni versiyonun genellikle
daha iyi bir Pareto eğrisi sunduğu, yani aynı hızda daha yüksek doğruluk
veya aynı doğrulukta daha yüksek hız sağladığı görülmektedir.

YOLO ile İnsan Tespiti ve Kişi Sayma
====================================

Bu bölümde, hazır eğitilmiş bir YOLO modeli kullanarak canlı video akışı
veya görüntüler üzerinde insan tespiti yapmayı ve tespit edilen kişileri
saymayı inceleyeceğiz. Sağlanan Python kodu üzerinden adım adım
ilerleyerek her bir satırın ne anlama geldiğini ve nasıl çalıştığını
açıklayacağız.
