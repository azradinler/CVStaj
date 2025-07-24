import numpy as np
from collections import deque

# İki kutunun kesişim bölgelerinin birleşim bölgelerine oranını hesaplayan yardımcı fonksiyon
def compute_iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    xi1 = max(x11, x21)
    yi1 = max(y11, y21)
    xi2 = min(x12, x22)
    yi2 = min(y12, y22)

    if xi2 <= xi1 or yi2 <= yi1:
        return 0

    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Takipçi sınıfı:
# iou_threshold: Bounding boxları eşlemek için gereken minimum oran.
# max_lost_frames: Takip edilen bounding boxun kaybolduğunda tamamen silinmeden önce hiç gözükmeden geçmesi gereken frame sayısı

class PersonTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=24):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        self.tracked_objects = {}
        self.total_unique_ids = set()

    def update_tracks(self, detections):
        current_frame_tracks = {}

        for track_id in list(self.tracked_objects.keys()):
            self.tracked_objects[track_id]['lost_frames'] += 1

            # Çok uzun süredir gözükmeyenleri sil
            if self.tracked_objects[track_id]['lost_frames'] > self.max_lost_frames:
                del self.tracked_objects[track_id]

        # detection: bu framede gelen bounding boxlar,
        # track: Bu nesnenin önceden takip ettiği bounding boxlar

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracked_objects.keys())
        matches = []

        # Bu kısım şunu yapıyor:
        # detection x track kadar bir matrix oluşturuyor, her bir detectionun her bir track ile IOU sunu hesaplıyor,
        # En yüksek IOU ya sahip olanları eşliyor, sonra onlara sıfır verip sonraki en çok eşleşenleri eşliyor,
        # Böylece önceki takip edilen bounding boxlar ile yeni tespit edilen bounding boxlar eşleşiyor (Eşleşmeyenler de olabilir)

        if len(detections) > 0 and len(unmatched_tracks) > 0:
            iou_matrix = np.zeros((len(detections), len(unmatched_tracks)))

            for d, detection in enumerate(detections):
                det_box = detection['box']
                for t, track_id in enumerate(unmatched_tracks):
                    track_box = self.tracked_objects[track_id]['box']
                    iou_matrix[d, t] = compute_iou(det_box, track_box)

            while True:
                max_iou = np.max(iou_matrix)

                # Eğer threshold değerinden büyük IOU ya sahip eşleşme yoksa bitir
                if max_iou < self.iou_threshold:
                    break

                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det_idx, track_idx = max_idx

                detection_idx = unmatched_detections[det_idx]
                track_id = unmatched_tracks[track_idx]

                # Eşleşmeleri kaydet
                matches.append((detection_idx, track_id))

                # Eşlenen değerlerin kutularını dahil etmeyi bırak.
                iou_matrix[det_idx, :] = 0
                iou_matrix[:, track_idx] = 0

        matched_detections = set()
        matched_tracks = set()

        for det_idx, track_id in matches:
            detection = detections[det_idx]

            # Yeni tespit ile eşleşen ve önceden takip edilen kutucuğun özelliklerini güncelle
            self.tracked_objects[track_id].update({
                'box': detection['box'],
                'confidence': detection['confidence'],
                'lost_frames': 0
            })
            current_frame_tracks[track_id] = self.tracked_objects[track_id]

            matched_detections.add(det_idx)
            matched_tracks.add(track_id)

        for det_idx in range(len(detections)):
            if det_idx not in matched_detections:

                # Eşleşmeyen yeni tespitler için yeni takip oluştur.
                detection = detections[det_idx]
                new_track = {
                    'id': self.next_id,
                    'box': detection['box'],
                    'confidence': detection['confidence'],
                    'lost_frames': 0
                }

                self.tracked_objects[self.next_id] = new_track
                current_frame_tracks[self.next_id] = new_track
                self.total_unique_ids.add(self.next_id)
                self.next_id += 1

        return current_frame_tracks, len(self.tracked_objects)




# 3 nokta arasındaki açıyı hesapla
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# Keypointlerden gelen bilgilere göre poz tahmini,
# Keypointler arası açılara ve mesafelere göre basit poz tahminleri, doğruluğu çok yüksek değil ayrıca bir sürü durumda yanlış sonuç verebilir

def classify_pose(keypoints, ankle_dist_history, box_width):
    for idx in [5, 6, 11, 12, 13, 14, 15, 16]:
        if np.allclose(keypoints[idx], [0, 0]):
            return "UNKNOWN"
    # Omuz, kalça, diz, bilek keypointlerinin aralarındaki açılar pozlar için belirleyici
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ankle, right_ankle = keypoints[15], keypoints[16]

    # Açıları hesapla
    left_body_leg_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_body_leg_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # omuz, kalça, diz açısı doğrusal değil de dike daha yakınsa oturuyor tahmini yapar
    if right_body_leg_angle < 130 and left_body_leg_angle < 130:
        return "SITTING"

    #Açıları hesapla
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # Ayakta durma koşulu oldukça katı, son 16 kare boyunca bieklerin birbirinden çok uzaklaşmamış olması gerekiyor,
    # Bel dik ve bacaklar da bükülmemiş pozisyonda olmalı, yine de çoğu durumda yetersiz ya da yanlış sonuç veriyor
    if ankle_dist_history is not None and len(ankle_dist_history) == 16:
        max_ankle = max(ankle_dist_history)
        min_ankle = min(ankle_dist_history)
        if box_width > 0 and (max_ankle - min_ankle) / box_width < 0.08:
            if (
                    left_body_leg_angle > 165
                    and right_body_leg_angle > 165
                    and right_knee_angle > 168
                    and left_knee_angle > 168
            ):
                return "STANDING"

    # Ne oturuyor ne ayakta duruyor ise diz açısına bakarak koşma ya da yürüme tahmininde bulunur
    # Koşma pozisyonunda diz açısının dikleştiği frameler olur.
    if right_knee_angle < 115 or left_knee_angle < 115:
        return "RUNNING"
    else:
        return "WALKING"

# PersonTracker sınıfının gelişmiş hali, +poz takibi yapıyor, geri kalan her şey aynı
class PoseTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=24):
        self.pose_history_len = 10
        self.ankle_history_len = 16
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        self.tracked_objects = {}
        self.total_unique_ids = set()

    def update_tracks(self, detections, keypoints_list):
        current_frame_tracks = {}

        for track_id in list(self.tracked_objects.keys()):
            self.tracked_objects[track_id]['lost_frames'] += 1

            # Çok uzun süredir gözükmeyenleri sil
            if self.tracked_objects[track_id]['lost_frames'] > self.max_lost_frames:
                del self.tracked_objects[track_id]

        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracked_objects.keys())
        matches = []

        #Person tracker sınıfındaki eşleştirme sisteminin aynısı
        if len(detections) > 0 and len(unmatched_tracks) > 0:
            iou_matrix = np.zeros((len(detections), len(unmatched_tracks)))

            for d, detection in enumerate(detections):
                det_box = detection['box']
                for t, track_id in enumerate(unmatched_tracks):
                    track_box = self.tracked_objects[track_id]['box']
                    iou_matrix[d, t] = compute_iou(det_box, track_box)
            while True:
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                det_idx, track_idx = max_idx

                detection_idx = unmatched_detections[det_idx]
                track_id = unmatched_tracks[track_idx]

                matches.append((detection_idx, track_id))

                iou_matrix[det_idx, :] = 0
                iou_matrix[:, track_idx] = 0

        matched_detections = set()
        matched_tracks = set()

        for det_idx, track_id in matches:
            detection = detections[det_idx]
            keypoints = keypoints_list[det_idx]
            obj = self.tracked_objects[track_id]
            x1, y1, x2, y2 = detection['box']
            box_width = abs(x2 - x1)
            ankle_dist = np.linalg.norm(np.array(keypoints[15]) - np.array(keypoints[16]))

            # Güncelle
            obj.update({
                'box': detection['box'],
                'lost_frames': 0,
                'keypoints': keypoints,
            })
            if 'pose_history' not in obj:
                obj['pose_history'] = deque(maxlen=self.pose_history_len)
            if 'ankle_dist_history' not in obj:
                obj['ankle_dist_history'] = deque(maxlen=self.ankle_history_len)
            obj['ankle_dist_history'].append(ankle_dist)

            # Poz tahmini
            pose = classify_pose(keypoints, obj['ankle_dist_history'], box_width)
            obj['pose_history'].append(pose)

            # Poz geçmişine pozlara bakarak nihai pozu tahmin et
            # Bunun yapmamın sebebi yürüme koşma gibi dinamik pozlar bazı karelerde standinge benziyor.
            ph = list(obj['pose_history'])
            stable_pose = None
            if ph.count('SITTING') >= 4:
                stable_pose = 'SITTING'
            elif ph.count('RUNNING') >= 2:
                stable_pose = 'RUNNING'
            elif ph.count('WALKING') >= 2:
                stable_pose = 'WALKING'
            elif ph.count('STANDING') >= 6:
                stable_pose = 'STANDING'
            obj['stable_pose'] = stable_pose
            current_frame_tracks[track_id] = obj
            matched_detections.add(det_idx)
            matched_tracks.add(track_id)

        for det_idx in range(len(detections)):
            if det_idx not in matched_detections:
                detection = detections[det_idx]
                keypoints = keypoints_list[det_idx]
                x1, y1, x2, y2 = detection['box']
                box_width = abs(x2 - x1)
                ankle_dist = np.linalg.norm(np.array(keypoints[15]) - np.array(keypoints[16]))
                pose = classify_pose(keypoints, None, box_width)
                new_track = {
                    'id': self.next_id,
                    'box': detection['box'],
                    'lost_frames': 0,
                    'keypoints': keypoints,
                    'pose_history': deque([pose], maxlen=self.pose_history_len),
                    'ankle_dist_history': deque([ankle_dist], maxlen=self.ankle_history_len),
                    'stable_pose': pose
                }
                current_frame_tracks[self.next_id] = new_track
                self.tracked_objects[self.next_id] = new_track
                self.total_unique_ids.add(self.next_id)
                self.next_id += 1
        return current_frame_tracks, len(self.tracked_objects)