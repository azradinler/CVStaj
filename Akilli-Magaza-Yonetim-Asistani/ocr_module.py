import cv2
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
import pytesseract
import csv
import json


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

video_source = 0 

model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Hata: Video kaynağına erişilemiyor.")
    exit()

person_data = defaultdict(lambda: {"position": "Bilinmiyor", "start_time": time.time(), "frame_start": 0, "ocr_text": ""})
position_history = defaultdict(list)
frame_rate = 30  

last_ocr_time = 0
ocr_interval = 5  

def get_person_position(keypoints):
    if len(keypoints) < 17:
        return "Bilinmiyor"
    
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
    ankle_left_x, ankle_left_y = keypoints[15]
    ankle_right_x, ankle_right_y = keypoints[16]
    
    hip_shoulder_distance = hip_y - shoulder_y
    
    hip_knee_distance = knee_y - hip_y
    
    ankle_horizontal_distance = abs(ankle_left_x - ankle_right_x)
    
    if hip_shoulder_distance < 100 and hip_knee_distance < 50:
        return "Oturuyor"
        
    if hip_shoulder_distance > 150 and ankle_horizontal_distance > 50:
        return "Yuruyor"
    
    if hip_shoulder_distance > 150:
        return "Ayakta"
    
    return "Bilinmiyor"

def save_to_csv(data, filename="behavior_log.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Pozisyon', 'Sure (saniye)', 'OCR Metni'])
        for track_id, history in data.items():
            for log in history:
                writer.writerow([track_id, log['position'], log['duration'], log.get('ocr_text', '')])

def save_to_json(data, filename="behavior_log.json"):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

start_time = time.time()
frame_count = 0

print("Akıllı Mağaza Yönetim Asistanı başlatılıyor...")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_frame_time = time.time()
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=0)

    for result in results:
        if result.keypoints is not None:
            for person_keypoints in result.keypoints.xy:
                for kp in person_keypoints:
                    x, y = map(int, kp)
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            track_id = int(box.id[0]) if box.id is not None else None
            
            if confidence > 0.5 and track_id is not None:
                keypoints_data = result.keypoints.xy[0].cpu().numpy()
                position = get_person_position(keypoints_data)
                
                if person_data[track_id]["position"] != position:
                    if person_data[track_id]["position"] != "Bilinmiyor":
                        duration_frames = frame_count - person_data[track_id]["frame_start"]
                        duration_seconds = duration_frames / frame_rate
                        position_history[track_id].append({
                            "position": person_data[track_id]["position"], 
                            "duration": duration_seconds,
                            "ocr_text": person_data[track_id]["ocr_text"]
                        })
                    
                    person_data[track_id]["position"] = position
                    person_data[track_id]["start_time"] = current_frame_time
                    person_data[track_id]["frame_start"] = frame_count
                    person_data[track_id]["ocr_text"] = "" #

                color = (0, 255, 0)
                label_text = f'ID: {track_id} | {position}'
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0] + 5, y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if position == "Ayakta" and (current_frame_time - last_ocr_time > ocr_interval):
                   
                    cropped_region = frame[y1 : y1 + (y2 - y1)//2, x1:x2]
                    
                    if cropped_region.shape[0] > 0 and cropped_region.shape[1] > 0:
                        gray_image = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
                        threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        
                        try:
                            ocr_text = pytesseract.image_to_string(threshold_image, lang='eng')
                            clean_ocr_text = '\n'.join(line for line in ocr_text.splitlines() if line.strip())
                            
                            if clean_ocr_text:
                                print(f"--- ID {track_id} Icin OKUNAN METIN ---")
                                print(clean_ocr_text)
                                print("-" * 30)
                                person_data[track_id]["ocr_text"] = clean_ocr_text
                                cv2.putText(frame, "Metin okundu!", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            last_ocr_time = current_frame_time
                        except Exception as e:
                            print(f"OCR hatasi: {e}")

    cv2.imshow("Akilli Magaza Yonetim Asistani", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

if position_history:
    save_to_csv(position_history)
    save_to_json(position_history)
    print("\nDavranış verileri 'behavior_log.csv' ve 'behavior_log.json' dosyalarına kaydedildi.")