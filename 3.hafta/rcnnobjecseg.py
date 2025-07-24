import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn#Mask R-CNN modeli (ResNet50 + FPN ile).
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random#Rastgele renk seçmek için (maskeler için).

# Modeli yükle (önceden eğitilmiş)
model = maskrcnn_resnet50_fpn(pretrained=True)# COCO veri seti üzerinde önceden eğitilmiş ağırlıkları kullan.
model.eval()

# Görüntüyü yükle
image_path = "pic1.jpg"  # Buraya kendi görselinin adını yaz
image = Image.open(image_path).convert("RGB")

# Görüntüyü hazırla
#ToTensor(): Görseli [0, 255] aralığından [0.0, 1.0] aralığına çevirip tensor formatına getirir.
#.unsqueeze(0): Modele vermek için bir batch dimension (1 x C x H x W) ekler.
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(image).unsqueeze(0)

# Tahmin yap
with torch.no_grad():
    prediction = model(img_tensor)#tahmin yapma kısmı

# Sonuçları al
pred_scores = prediction[0]['scores'].detach().numpy()
pred_boxes = prediction[0]['boxes'].detach().numpy()
pred_labels = prediction[0]['labels'].detach().numpy()
pred_masks = prediction[0]['masks'].detach().numpy()

# Eşik belirle
threshold = 0.8

# Orijinal resmi kopyala
image_draw = image.copy()
draw = ImageDraw.Draw(image_draw)

# Font
try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

for i in range(len(pred_scores)):
    if pred_scores[i] > threshold:
        box = pred_boxes[i]
        label = pred_labels[i]
        mask = pred_masks[i, 0]

        # Kutu çiz
        draw.rectangle(box.tolist(), outline="red", width=2)
        draw.text((box[0], box[1] - 10), f"Class: {label}", fill="yellow", font=font)

        # Maskeyi göster (şeffaf)
        mask_img = Image.fromarray((mask > 0.5).astype("uint8") * 255)
        color = random.randint(100, 255)
        mask_overlay = Image.new("RGBA", image.size, (color, 0, 0, 100))
        image_draw.paste(mask_overlay, (0, 0), mask_img)

# Sonucu göster
plt.imshow(image_draw)
plt.title("Mask R-CNN Object Segmentation")
plt.axis("off")
plt.show()
