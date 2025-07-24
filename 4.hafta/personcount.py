import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"# duplicate library hatalarını önlemek için ortam değişkeni ayarlar.

from ultralytics import YOLO
import cv2
from sort import Sort#SORT (Simple Online and Realtime Tracking) algoritması — tespit edilen nesneleri takip etmek için.
import numpy as np

model = YOLO("yolov8n.pt")
tracker = Sort()

cap = cv2.VideoCapture("people-detection.mp4")#kare kare video okur
counted_ids = set()#Takip edilen ve sayılmış kişilerin ID'lerini tutar. Aynı kişiyi birden fazla saymamak için set kullanılır.
track_stability = {}  # Yeni: ID -> kare sayısı

while True:#video karelerinin okunma döngüsü
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]#o anki kareden alınan sonuç
    detections = []

    for result in results.boxes:#insan tespiti
        cls = int(result.cls)
        if cls == 0:  # sadece insan sınıfı
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            conf = float(result.conf)
            detections.append([x1, y1, x2, y2, conf])

    if len(detections) > 0:#bir insan tespit edilirse id atanır
        track_results = tracker.update(np.array(detections))
    else:
        track_results = np.empty((0, 5))

    for trk in track_results:
        x1, y1, x2, y2, track_id = map(int, trk)

        track_stability[track_id] = track_stability.get(track_id, 0) + 1
        if track_stability[track_id] == 3:  # 3 kare boyunca görüldüyse say
            counted_ids.add(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f'Total People: {len(counted_ids)}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Crowd Tracking", frame)

    # Kapanma kontrolü
    if cv2.getWindowProperty("Crowd Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
