import cv2
from ultralytics import YOLO
from persontracker import PoseTracker


def resize_with_aspect_ratio(image, max_width=1600, max_height=800):
    h, w = image.shape[:2]
    width_ratio = max_width / w
    height_ratio = max_height / h
    scale_ratio = min(width_ratio, height_ratio)
    new_width = int(w * scale_ratio)
    new_height = int(h * scale_ratio)
    return cv2.resize(image, (new_width, new_height))

def main():
    video_path = "input_video.mp4"
    model = YOLO("yolov8n-pose.pt")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Video dosyası açılamadı")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video özellikleri: {width}x{height}, {fps:.2f} FPS, {total_frames} frame")

    # Pose trackeri başlat, person tracker ile neredeyse aynı şekilde kullanılıyor, sadece update fonksiyonuna keypointler gönderilmeli
    tracker = PoseTracker(iou_threshold=0.3, max_lost_frames=24)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_with_aspect_ratio(frame)
        frame_count += 1

        results = model(frame, conf=0.6, iou=0.4, verbose=False)

        keypoints_list = results[0].keypoints.xy.tolist() if results[0].keypoints is not None else []
        boxes = results[0].boxes.xyxy.tolist() if results[0].boxes is not None else []
        detections = [{'box': tuple(box)} for box in boxes]

        current_tracks, active_person = tracker.update_tracks(detections, keypoints_list)
        for track_id, track in current_tracks.items():
            x1, y1, x2, y2 = track['box']
            keypoints = track['keypoints']
            stable_pose = track['stable_pose']

            if stable_pose:
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                for (x, y) in keypoints:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                text = f"ID:{track_id} {stable_pose}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"Aktif Kisi: {active_person}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Poz Takibi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()