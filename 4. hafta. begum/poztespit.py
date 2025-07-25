import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("D:/sistem/Videolar/babacocuk.mp4")

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
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

        # Kırpılan kişi görüntüsü
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            continue

        img_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)

        if res.pose_landmarks:
            mp_draw.draw_landmarks(person_img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lm = res.pose_landmarks.landmark
            h, w = person_img.shape[:2]

            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

            angle = calculate_angle(hip, knee, ankle)

            if angle < 90:
                status = "Diz Cokmus / Oturuyor"
                color = (0, 255, 0)
            else:
                status = "Ayakta"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Baba ve Oğul Poz Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
