import cv2
from ultralytics import YOLO

# YOLOv8 modelini yükle (n = nano, hızlı ve küçük)
model = YOLO("yolov8n.pt")

# Video dosyasını yükle
video_path = r"D:\sistem\Videolar\square.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO ile tespit yap
    results = model(frame)[0]

    person_count = 0

    # Tespit edilen kutular üzerinden geç
    for box in results.boxes:
        cls_id = int(box.cls[0])         # sınıf indexi (int)
        conf = float(box.conf[0])        # güven skoru
        if model.names[cls_id] == 'person':  # sadece kişi sınıfını say
            person_count += 1

            # Kutu koordinatlarını al ve çiz
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Kişi sayısını ekranda göster
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # EKRANI KÜÇÜLT: (640x360 boyutuna getiriyoruz)
    frame = cv2.resize(frame, (960, 540))

    # Kareyi göster
    cv2.imshow("YOLOv8 Person Detection", frame)

    # 'q' tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

