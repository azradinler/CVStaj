from ultralytics import YOLO#insan tespiti yapan model
import cv2
import mediapipe as mp#google tarafından geliştirilmiş keypointi tesiti yapan kütüphane 

person_detector = YOLO("yolov8n.pt")   # YOLOv8 Nano modeli (hafif versiyon)
mp_pose = mp.solutions.pose            # MediaPipe Pose modülü (sınıf)
pose = mp_pose.Pose()                  # Pose sınıfından nesne (varsayılan ayarlar)

cap = cv2.VideoCapture("people-detection.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # İnsan tespiti
    results = person_detector(frame)[0]#box bilgilerini kaydediyor

    for box in results.boxes:
        if int(box.cls) == 0:  # sadece insan sınıfı
            x1, y1, x2, y2 = map(int, box.xyxy[0])#kordinatları aldık
            person_crop = frame[y1:y2, x1:x2]#kırpıyoruz

            # Pose tahmini
            person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            result = pose.process(person_rgb)#tahmin yapılıyor

            if result.pose_landmarks:#noktaları yerine koyuyoruz
                for lm in result.pose_landmarks.landmark:
                    cx = int(lm.x * (x2 - x1)) + x1
                    cy = int(lm.y * (y2 - y1)) + y1
                    cv2.circle(frame, (cx, cy), 3, (0,255,0), -1)

    cv2.imshow("Top-Down Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
