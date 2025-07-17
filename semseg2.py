from torchvision import models, transforms#Önceden eğitilmiş modelleri içerir (burada DeepLabV3) ve uygun forma getirme
import torch#PyTorch'un kendisi.
from PIL import Image
import matplotlib.pyplot as plt

# Modeli yükle
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Görseli hazırla
img = Image.open("city2.jpg").convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize(520),#kısa kenar
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])# Görseli ImageNet ortalama ve std değerleriyle normalize eder (model bu şekilde eğitilmiştir).
input_tensor = preprocess(img).unsqueeze(0)
#.unsqueeze(0) ile batch dimension (1, C, H, W) eklenir → modele uygun hale gelir.
# Segmentasyon yap
# sonucu bir sözlük döndürür: 'out' anahtarı segmentasyon skorlarını içerir.
#[0] → batch'teki ilk (ve tek) örneği alırız.
with torch.no_grad():
    output = model(input_tensor)['out'][0]
#argmax(0) → Her pikselde en yüksek skoru alan sınıfı bulur.
#.byte() → uint8'e dönüştürülür.
#.cpu().numpy() → CPU’ya alınır ve NumPy array’e çevrilir (görselleştirme için).
seg = output.argmax(0).byte().cpu().numpy()

# Göster
plt.imshow(seg)
plt.title("Segmentasyon Maskesi")
plt.axis('off')
plt.show()
