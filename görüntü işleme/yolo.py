import numpy as np
import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    while True:
        ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    'YOLO ile Nesne Tespiti'

 results=model(frame)     
    boxes = results[0].boxes.xyxy
    labels = results[0].boxes.cls
    confidences = results[0].boxes.conf

    'İnsan Tespiti ve Sayma'
person_count=0
    for box, label, confidence in zip(boxes, labels, confidences):
        x1, y1, x2, y2 = box
        label = int(label)
        snf=model.names[label]
        confidence = float(confidence)
        if snf=="person":
            person_count=person_count+1
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            text = f'{person_count} ---- {snf} {confidence:.2f}'
            cv.putText(frame, text, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    'Sonuçları Görüntüleme ve Çıkış'
cv.putText(frame, f"toplam insan:{person_count}", (400, 20 ), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)        
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()