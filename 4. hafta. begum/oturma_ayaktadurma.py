import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import math

# Pose Estimation (MediaPipe)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

# İnsan algılayıcı (YOLOv8)
model = YOLO("yolov8n.pt")  # Küçük ve hızlı versiyon

# Video yükle
cap = cv2.VideoCapture("D:/sistem/Videolar/oturma.mp4")

def calculate_angle(a, b, c):
    """Üç nokta arasındaki açıyı hesaplar (diz açısı için)"""
    a = np.array(a)  # kalça
    b = np.array(b)  # diz
    c = np.array(c)  # ayak bileği

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    results = model(frame)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != 'person':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue

        img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            try:
                # Sol bacak
                hip_l = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee_l = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle_l = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle_l = calculate_angle(hip_l, knee_l, ankle_l)

                # Sağ bacak
                hip_r = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                knee_r = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ankle_r = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                angle_r = calculate_angle(hip_r, knee_r, ankle_r)

                # Ortalama açı veya min açıya göre karar
                min_angle = min(angle_l, angle_r)

                if min_angle < 100:  # Daha keskin bir eşik değeri
                    status = "Oturuyor"
                    renk = (0, 255, 0)
                else:
                    status = "Ayakta"
                    renk = (0, 0, 255)

                # Görselleştir
                cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(frame, f"{status} ({int(min_angle)}°)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)

            except Exception as e:
                print("Pose işlem hatası:", e)
                continue

    cv2.imshow("Pose ile Oturan vs Ayakta", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
