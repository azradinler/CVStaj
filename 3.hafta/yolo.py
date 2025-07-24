from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# Hazır eğitilmiş YOLOv8 modeli (COCO veri setine göre eğitildi)
model = YOLO("yolov8n.pt")  # n: nano, s: small, m, l, x → hız/başarı değişir

# Görüntüyü yükle
image = cv2.imread("pic1.jpg")

# YOLO, RGB formatta çalışır
results = model.predict(source=image, conf=0.3)

# Sonuçları görselleştir
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, model.names[cls_id], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("people detection")
plt.axis("off")
plt.show()
