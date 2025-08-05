import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
import csv
import json

video_source = 0 

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Hata: Video kaynağına erişilemiyor.")
    exit()

person_positions = {}
position_history = defaultdict(list)
frame_rate = 30 

def get_person_position(keypoints):
    if len(keypoints) < 17:
        return "Bilinmiyor"
   
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
    ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
    
    if hip_y - shoulder_y < 100 and knee_y - hip_y < 50:
        return "Oturuyor"
        
    if abs(keypoints[15][0] - keypoints[16][0]) > 50:
        return "Yuruyor"
    
    return "Ayakta"
    
start_time = time.time()
frame_count = 0

def save_to_csv(data, filename="behavior_log.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Pozisyon', 'Sure (saniye)'])
        for track_id, history in data.items():
            for log in history:
                writer.writerow([track_id, log['position'], log['duration']])

def save_to_json(data, filename="behavior_log.json"):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    current_frame_time = time.time()
    
    for result in results:
        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            confidence = box.conf[0]
            
            if confidence > 0.5 and track_id is not None:
                
                keypoints_data = result.keypoints.xy[0].cpu().numpy()
                position = get_person_position(keypoints_data)
                
                if track_id not in person_positions:
                    person_positions[track_id] = {"position": position, "start_time": current_frame_time, "frame_start": frame_count}
                
                if person_positions[track_id]["position"] != position:
                    duration_frames = frame_count - person_positions[track_id]["frame_start"]
                    duration_seconds = duration_frames / frame_rate
                    position_history[track_id].append({"position": person_positions[track_id]["position"], "duration": duration_seconds})
                    
                    person_positions[track_id]["position"] = position
                    person_positions[track_id]["start_time"] = current_frame_time
                    person_positions[track_id]["frame_start"] = frame_count

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                text_size, _ = cv2.getTextSize(f'ID: {track_id} | {position}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0] + 5, y1), color, -1)
                cv2.putText(frame, f'ID: {track_id} | {position}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        if result.keypoints is not None:
            for person_keypoints in result.keypoints.xy:
                for kp in person_keypoints:
                    x, y = map(int, kp)
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    cv2.imshow("YOLOv8 Davranis Analizi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

if position_history:
    save_to_csv(position_history)
    save_to_json(position_history)
    print("\nDavranış verileri 'behavior_log.csv' ve 'behavior_log.json' dosyalarına kaydedildi.")