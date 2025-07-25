import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_draw = mp.solutions.drawing_utils

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("D:/sistem/Videolar/ders.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    oturan_sayisi = 0
    ayakta_sayisi = 0
    konusmaci_id = None

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == 'person':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0:
                continue

            img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            if res.pose_landmarks:
                landmarks = res.pose_landmarks.landmark

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * (x2 - x1),
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * (y2 - y1)]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * (x2 - x1),
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * (y2 - y1)]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * (x2 - x1),
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * (y2 - y1)]

                angle = calculate_angle(left_hip, left_knee, left_ankle)

                if angle < 140:
                    pozisyon = "Dinleyici (Oturan)"
                    oturan_sayisi += 1
                    renk = (0, 255, 0)
                else:
                    pozisyon = "Konuşmacı (Ayakta)"
                    ayakta_sayisi += 1
                    renk = (0, 0, 255)

                # Kutu çiz ve pozisyon yaz
                cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(frame, pozisyon, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)

    cv2.putText(frame, f"Dinleyici Sayisi: {oturan_sayisi}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Konusmaci Sayisi: {ayakta_sayisi}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Otobus Konusmaci ve Dinleyici Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
