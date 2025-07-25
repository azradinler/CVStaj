import cv2
import mediapipe as mp
import numpy as np

# Pose Estimation ayarları
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

cap = cv2.VideoCapture("D:/sistem/Videolar/spor.mp4")  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        lm = results.pose_landmarks.landmark
        height, width = frame.shape[:2]

        # Sol diz açısı
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
               lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height]

        angle = calculate_angle(hip, knee, ankle)

        # Basit sınıflandırma
        if angle < 100:
            status = "Oturuyor"
            color = (0, 255, 0)
        else:
            status = "Ayakta / Kosuyor"
            color = (0, 0, 255)

        cv2.putText(frame, f"Diz Aci: {int(angle)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, status, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Basit Hareket Ayrimi", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
