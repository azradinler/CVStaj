import cv2
from ultralytics import YOLO
from persontracker import PersonTracker

# Görselleri ekrana sığacak hale getirir
def resize_with_aspect_ratio(image, max_width=1600, max_height=800):
    h, w = image.shape[:2]
    width_ratio = max_width / w
    height_ratio = max_height / h
    scale_ratio = min(width_ratio, height_ratio)
    new_width = int(w * scale_ratio)
    new_height = int(h * scale_ratio)
    return cv2.resize(image, (new_width, new_height))


def main():
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture("input_video.mp4") # Video seç

    if not cap.isOpened():
        print("Video dosyası açılamadı")
        return

    # Video özellikleri
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video özellikleri: {width}x{height}, {fps:.2f} FPS, {total_frames} frame")

    tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=30)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_with_aspect_ratio(frame)
        frame_count += 1

        # conf: Tahmin edilen nesnenin güven skoru en az %60 olacak,daha düşük skorlar filtrelenir.
        # iou : Kutu örtüşme oranı %40’tan fazla ise sadece en yüksek skora sahip olan kutu tutulur.
        # verbose = False: Konsola bilgileri yazdırmaz

        results = model(frame, conf=0.6, iou=0.4, verbose=False)
        boxes = results[0].boxes  # Tespit edilen kutular(bounding box)
        names = results[0].names  # Tespit edilen sınıf isimleri

        detections = []

        if boxes is not None:
            for i in range(len(boxes.cls)):
                class_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())

                # İnsan değilse yok say
                if names[class_id] != "person":
                    continue

                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                # Camdaki yansımayı yok say (sadece örnekteki videoya özel, başka videolarda başka koordinatlarda gerekebilir)
                #if x1 <= 700 <= x2 and y1 <= 518 <= y2:
                    #continue

                # Çok küçük kutucukları yok say
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width < 30 or box_height < 50:
                    continue

                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence
                })

        # Person tracker nesnesine bu framedeki bilgileri gönder.
        current_tracks, active_person = tracker.update_tracks(detections)

        # Görselleştirme
        for track_id, track in current_tracks.items():
            x1, y1, x2, y2 = track['box']
            confidence = track['confidence']

            # Kutu çizme
            color = (0, 255, 0) if track['lost_frames'] == 0 else (0, 165, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # ID ve confidence yazma
            label = f"ID:{track_id} ({confidence:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)

        # İstatistikleri yazdır
        cv2.putText(frame, f"Aktif Kisi: {active_person}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Toplam Kisi: {len(tracker.total_unique_ids)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Sayma", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Toplam görülen kişi: {len(tracker.total_unique_ids)}")


if __name__ == "__main__":
    main()