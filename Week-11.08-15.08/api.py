import asyncio
from typing import Optional
import csv
import os
from datetime import datetime, date

import cv2
import torch
import numpy as np
from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import time
import gc


class PeopleCounterService:
    def __init__(self) -> None:
        self._entry_count = 0
        self._exit_count = 0
        self._is_running = False
        self._latest_frame_jpeg: Optional[bytes] = None
        self._latest_heatmap_jpeg: Optional[bytes] = None
        self._task: Optional[asyncio.Task] = None

        torch.backends.cudnn.benchmark = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tracker = None
        self.cap = None
        self.new_width = 0
        self.new_height = 0

        self.region_x1 = 0
        self.region_y1 = 0
        self.region_x2 = 0
        self.region_y2 = 0

        self.track_memory = {}

        self.DETECTION_INTERVAL = 3
        self.HEATMAP_INTERVAL = 10
        self.DISPLAY_INTERVAL = 5
        self.cleanup_interval = 100

        self.heatmap_scale = 8
        self.heatmap_height = 0
        self.heatmap_width = 0
        self.heatmap = None

        self.last_detections = []
        self.cached_heatmap_overlay = None
        self.last_event_time = None

        self.fps_times = deque(maxlen=10)
        self.frame_count = 0
        self.current_fps = 0

        self.reports_dir = "reports"
        self.ensure_reports_directory()
        self.current_date = date.today()
        self.csv_filename = None
        self.csv_file = None
        self.csv_writer = None
        self.initialize_csv()

    def ensure_reports_directory(self):
        if not os.path.exists(self.reports_dir):
            os.makedirs(self.reports_dir)
            print(f"Reports directory created: {self.reports_dir}")

    def initialize_csv(self):
        self.current_date = date.today()
        self.csv_filename = os.path.join(self.reports_dir, f"people_count_{self.current_date.strftime('%Y-%m-%d')}.csv")

        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Event', 'Track_ID', 'Total_Entry', 'Total_Exit', 'Current_Count'])
                print(f"Yeni CSV dosyası oluşturuldu: {self.csv_filename}")

    def log_event(self, event_type: str, track_id: int):
        try:
            if date.today() != self.current_date:
                self.initialize_csv()

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_count = self._entry_count - self._exit_count

            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, event_type, track_id, self._entry_count, self._exit_count, current_count])

            print(f"LOG: {timestamp} - {event_type} - Track ID: {track_id}")

        except Exception as e:
            print(f"Loglarken hata oluştu: {e}")

    async def start(self, video_path: str = "Data/large.mp4") -> bool:
        if self._is_running:
            return False

        self.model = YOLO("yolov8s.pt").to(self.device)
        self.model.fuse()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"Video özellikleri: {width}x{height}, {fps:.2f} FPS")

        w_ratio = 1600 / width
        h_ratio = 800 / height
        s_ratio = min(w_ratio, h_ratio)
        self.new_width = int(width * s_ratio)
        self.new_height = int(height * s_ratio)

        self.tracker = DeepSort(
            max_age=20,
            n_init=2,
            max_cosine_distance=0.4,
            nn_budget=50
        )

        self.region_x1 = int(self.new_width * 0.25)
        self.region_y1 = int(self.new_height * 0.25)
        self.region_x2 = int(self.new_width * 0.75)
        self.region_y2 = int(self.new_height * 0.75)

        self.heatmap_height = max(self.new_height // self.heatmap_scale, 50)
        self.heatmap_width = max(self.new_width // self.heatmap_scale, 50)
        self.heatmap = np.zeros((self.heatmap_height, self.heatmap_width), dtype=np.float32)

        self._entry_count = 0
        self._exit_count = 0
        self.track_memory = {}
        self.last_detections = []
        self.cached_heatmap_overlay = None
        self.last_event_time = None
        self.fps_times.clear()
        self.frame_count = 0

        self.initialize_csv()

        self._is_running = True
        self._task = asyncio.create_task(self._run_loop())
        return True

    async def stop(self) -> bool:
        if not self._is_running:
            return False

        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self.cap:
            self.cap.release()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return True

    def get_counts(self) -> dict:
        return {
            "entry": int(self._entry_count),
            "exit": int(self._exit_count),
            "running": bool(self._is_running),
            "fps": float(self.current_fps),
            "frame_count": int(self.frame_count)
        }

    def get_latest_frame(self) -> Optional[bytes]:
        return self._latest_frame_jpeg

    def get_latest_heatmap(self) -> Optional[bytes]:
        return self._latest_heatmap_jpeg

    def get_reports_list(self) -> list:
        try:
            if not os.path.exists(self.reports_dir):
                return []

            files = []
            for file in os.listdir(self.reports_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.reports_dir, file)
                    file_size = os.path.getsize(file_path)
                    files.append({
                        "filename": file,
                        "size": file_size,
                        "path": file_path
                    })
            return sorted(files, key=lambda x: x["filename"], reverse=True)
        except Exception as e:
            print(f"Error listing reports: {e}")
            return []

    async def _run_loop(self) -> None:
        try:
            while self._is_running:
                loop_start = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    break

                self.frame_count += 1
                frame = cv2.resize(frame, (self.new_width, self.new_height))

                if self.frame_count % self.DETECTION_INTERVAL == 0:
                    with torch.no_grad():
                        results = self.model.predict(
                            frame,
                            conf=0.3,
                            iou=0.5,
                            classes=[0],
                            verbose=False,
                            half=True,
                            device=self.device,
                            agnostic_nms=True
                        )[0]

                    detections = []
                    if results.boxes is not None and len(results.boxes) > 0:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        scores = results.boxes.conf.cpu().numpy()

                        if len(boxes) > 20:
                            top_indices = np.argsort(scores)[-20:]
                            boxes = boxes[top_indices]
                            scores = scores[top_indices]

                        for i, (box, score) in enumerate(zip(boxes, scores)):
                            x1, y1, x2, y2 = box
                            w, h = x2 - x1, y2 - y1
                            detections.append(([float(x1), float(y1), float(w), float(h)], float(score), "person"))

                    self.last_detections = detections
                else:
                    detections = self.last_detections

                tracks = self.tracker.update_tracks(detections, frame=frame)

                if self.frame_count % self.cleanup_interval == 0:
                    active_track_ids = {track.track_id for track in tracks if track.is_confirmed()}
                    self.track_memory = {tid: data for tid, data in self.track_memory.items()
                                        if tid in active_track_ids}

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    bbox = track.to_ltrb()
                    l, t, r, b = bbox
                    cx = int((l + r) / 2)
                    cy = int((t + b) / 2)

                    if self.frame_count % self.HEATMAP_INTERVAL == 0:
                        heatmap_x = min(max(0, cx // self.heatmap_scale), self.heatmap_width - 1)
                        heatmap_y = min(max(0, cy // self.heatmap_scale), self.heatmap_height - 1)
                        self.heatmap[heatmap_y, heatmap_x] += 5.0

                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                ny, nx = heatmap_y + dy, heatmap_x + dx
                                if 0 <= ny < self.heatmap_height and 0 <= nx < self.heatmap_width:
                                    distance = np.sqrt(dx*dx + dy*dy)
                                    if distance <= 2:
                                        self.heatmap[ny, nx] += 2.0 / (1 + distance)

                        if self.frame_count % (self.HEATMAP_INTERVAL * 3) == 0:
                            self.heatmap *= 0.90

                    inside_now = (self.region_x1 <= cx <= self.region_x2) and (self.region_y1 <= cy <= self.region_y2)

                    if track_id in self.track_memory:
                        prev_inside = self.track_memory[track_id][2]

                        if not prev_inside and inside_now:
                            self._entry_count += 1
                            self.log_event("ENTRY", track_id)
                            now = time.time()
                            if self.last_event_time is not None:
                                print(f"Giriş: {now - self.last_event_time:.2f} saniye sonra")
                            self.last_event_time = now

                        elif prev_inside and not inside_now:
                            self._exit_count += 1
                            self.log_event("EXIT", track_id)
                            now = time.time()
                            if self.last_event_time is not None:
                                print(f"Çıkış: {now - self.last_event_time:.2f} saniye sonra")
                            self.last_event_time = now

                    self.track_memory[track_id] = (cx, cy, inside_now)

                    cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

                cv2.rectangle(frame, (self.region_x1, self.region_y1), (self.region_x2, self.region_y2), (255, 0, 0), 2)

                loop_time = time.time() - loop_start
                self.fps_times.append(1.0 / loop_time if loop_time > 0 else 0)
                self.current_fps = sum(self.fps_times) / len(self.fps_times)

                ok_jpg, enc = cv2.imencode(".jpg", frame)
                if ok_jpg:
                    self._latest_frame_jpeg = enc.tobytes()

                if self.frame_count % self.DISPLAY_INTERVAL == 0:
                    heatmap_display = np.ones((self.new_height, self.new_width, 3), dtype=np.uint8) * 255
                    heatmap_resized = cv2.resize(self.heatmap, (self.new_width, self.new_height), interpolation=cv2.INTER_LINEAR)
                    heatmap_normalized = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    heatmap_normalized = np.power(heatmap_normalized / 255.0, 0.7) * 255
                    heatmap_normalized = heatmap_normalized.astype(np.uint8)

                    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

                    heatmap_display = cv2.addWeighted(heatmap_display, 0.1, heatmap_colored, 0.9, 0)
                    ok_heatmap, enc_heatmap = cv2.imencode(".jpg", heatmap_display)
                    if ok_heatmap:
                        self._latest_heatmap_jpeg = enc_heatmap.tobytes()

                await asyncio.sleep(0.001)

        except Exception as e:
            print(f"Error in run loop: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self._is_running = False


app = FastAPI(title="Optimized People Counter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartRequest(BaseModel):
    video_path: Optional[str] = None


service = PeopleCounterService()


@app.get("/")
async def root():
    return FileResponse("index.html")


@app.get("/counts")
def counts() -> dict:
    return service.get_counts()


@app.post("/start")
async def start(request: StartRequest) -> dict:
    started = await service.start(request.video_path or "Data/large.mp4")
    return {"started": started, **service.get_counts()}


@app.post("/stop")
async def stop() -> dict:
    stopped = await service.stop()
    return {"stopped": stopped, **service.get_counts()}


@app.get("/frame")
def frame() -> Response:
    jpeg = service.get_latest_frame()
    if not jpeg:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/heatmap")
def heatmap() -> Response:
    jpeg = service.get_latest_heatmap()
    if not jpeg:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


@app.get("/video")
async def video() -> StreamingResponse:
    boundary = "frame"

    async def mjpeg_stream():
        while True:
            jpeg = service.get_latest_frame()
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        mjpeg_stream(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/heatmap-stream")
async def heatmap_stream() -> StreamingResponse:
    boundary = "frame"

    async def mjpeg_stream():
        while True:
            jpeg = service.get_latest_heatmap()
            if jpeg:
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        mjpeg_stream(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


@app.get("/reports")
async def get_reports_list():
    reports = service.get_reports_list()
    return {"reports": reports}


@app.get("/reports/{filename}")
async def download_report(filename: str):
    filepath = os.path.join(service.reports_dir, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename)
    return {"error": "File not found"}


@app.get("/video-stream")
async def video_stream():
    jpeg = service.get_latest_frame()
    if not jpeg:
        return Response(status_code=204)
    return Response(content=jpeg, media_type="image/jpeg")


