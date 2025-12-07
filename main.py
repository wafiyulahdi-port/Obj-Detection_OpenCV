import os
import time
import cv2
import asyncio
import csv
from datetime import datetime
from src.detection import FaceDetector
from src.embedding import ArcFaceEmbedder
from src.recognition import FaceRecognizer
from src.blink import BlinkDetector 

# ================== CONFIG ==================
USE_FLIP = False          # True untuk kamera depan/selfie
DOWNSCALE = 0.75          # 1.0 = off; 0.5 lebih cepat tapi akurasi turun
KEEP_ALL = True           # False untuk wajah utama
THRESH = 0.90             # confidence threshold untuk deteksi wajah
DEVICE = None             # None = auto pilih CUDA/CPU
COOLDOWN = 2 * 60 * 60    # 2 jam dalam detik
ATT_FILE = "attendance.csv"
# ============================================

# Inisialisasi 
detector = FaceDetector(
    device=DEVICE,
    threshold=THRESH,
    keep_all=KEEP_ALL,
    downscale=DOWNSCALE,
    do_flip=USE_FLIP
)

embedder = ArcFaceEmbedder(
    model_root="models",
    model_name="arcface_r100_v1",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

recognizer = FaceRecognizer(
    threshold=0.80, 
    db_path="output/db_faces.npz", 
    data_path="data/employees"
)

blink_detector = BlinkDetector(ear_threshold=0.25, consecutive_frames=3)

# Load Database
recognizer.load_database()
if not recognizer.db_names:
    recognizer.build_from_images(embedder)

# Warm up
import numpy as np
dummy = np.zeros((112, 112, 3), dtype=np.uint8)      
dummy_emb = embedder.embed(dummy)                   
_ = recognizer.recognize(dummy_emb)                 

# Attendance Helper
def init_csv(file):
    if not os.path.exists(file):
        with open(file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "date", "check_in_time", "check_out_time", "status"])

def log_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M")
    now_sec = time.time()

    rows = []
    found = False
    updated = False

    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    # Cek log hari ini
    for row in rows:
        if row["date"] == today:
                if row["name"] == name:
                    found= True
                    if not row["check_in_time"]:
                        row["check_in_time"] = now_time
                        row["status"] = "Hadir"
                        updated = True

                    elif not row["check_out_time"]:
                        check_in_dt = datetime.strptime(row["check_in_time"], "%H:%M")
                        now_dt = datetime.strptime(now_time, "%H:%M")
                        elapsed = (now_dt - check_in_dt).total_seconds()
                        if elapsed >= 2 * 3600:  # minimal 2 jam setelah check in
                            row["check_out_time"] = now_time
                            row["status"] = "Hadir (Lengkap)"
                            updated = True
                    break

    if not found:
        rows.append({
            "name": name,
            "date": today,
            "check_in_time": now_time,
            "check_out_time": "",
            "status": "Hadir"
        })
        updated = True

    if updated:
        with open(ATT_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "date", "check_in_time", "check_out_time", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name} attendance updated")

# Buat file CSV kalau belum ada
init_csv(ATT_FILE)

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    raise SystemExit

fps_avg = 0.0
alpha = 0.9 

print("[Tekan 's' untuk simpan wajah ke database")
print("[Tekan 'q' untuk keluar")

while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        print("Gagal membaca frame")
        break

    # Default kamera selfie (flip horizontal)
    if detector.do_flip:
        frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]

    # Blink Detection
    frame, blinked, total_blinks = blink_detector.process_frame(frame)

    # Face Detection
    faces = detector.detect_faces(frame)
    vis = detector.draw_faces(frame, faces)

    if faces and faces[0].get('aligned') is not None:

        aligned = faces[0]['aligned']

        aligned_small = cv2.resize(aligned, (80, 80), interpolation=cv2.INTER_LINEAR)
        ah, aw = aligned_small.shape[:2]
        x_start = (W - aw) // 2
        y_start = 0
        vis[y_start:y_start+ah, x_start:x_start+aw] = aligned_small

        name, score = "Unknown", 0.0

        # Recognition hanya jika minimal 3 kali blink
        if total_blinks > 2:
            emb = embedder.embed(aligned)
            if emb is not None:
                name, score = recognizer.recognize(emb)

                if name != "Unknown":
                    log_attendance(name)

        cv2.putText(vis, f"{name} ({score:.2f})", (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

        if total_blinks < 3:
            cv2.putText(vis, "Harap kedip untuk verifikasi", (10, H-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1, cv2.LINE_AA)


    # FPS
    dt = time.time() - t0
    fps_inst = 1.0 / max(dt, 1e-6)
    fps_avg = alpha * fps_avg + (1 - alpha) * fps_inst if fps_avg > 0 else fps_inst

    cv2.putText(vis, f"FPS: {fps_avg:.1f}", (W-90, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

    cv2.imshow("Face Recognition + Blink", vis)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        detector.do_flip = not detector.do_flip
        print(f"Flip: {detector.do_flip}")
    elif key == ord('m'):
        detector.keep_all = not detector.keep_all
        print(f"Keep all faces: {detector.keep_all}")
    elif key == ord('s') and faces and faces[0].get('aligned') is not None and total_blinks > 0:

        # Simpan wajah ke database hanya kalau sudah blink
        emb = embedder.embed(faces[0]['aligned'])
        if emb is not None:
            name = input("Masukkan nama data baru: ")
            recognizer.add_identity(name, emb)
            recognizer.save_database()
            print(f"Wajah {name} ditambahkan ke database!")

cap.release()
cv2.destroyAllWindows()
