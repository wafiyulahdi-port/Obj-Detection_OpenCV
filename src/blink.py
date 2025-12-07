import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

class BlinkDetector:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        # Threshold EAR untuk dianggap blink
        self.EAR_THRESHOLD = ear_threshold
        self.CONSEC_FRAMES = consecutive_frames

        # Counter blink
        self.counter = 0
        self.total_blinks = 0

        # Inisialisasi Mediapipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Landmark index untuk mata kiri & kanan (EAR)
        # Sesuai Mediapipe (468 points)
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # [left, top1, top2, right, bottom1, bottom2]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    def _eye_aspect_ratio(self, eye_points):
        # Vertical
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        # Horizontal
        C = dist.euclidean(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def process_frame(self, frame):
        """Deteksi blink dari frame kamera."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        blink_detected = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Ambil koordinat mata
                left_eye = [(int(face_landmarks.landmark[i].x * w),
                             int(face_landmarks.landmark[i].y * h)) for i in self.LEFT_EYE_IDX]
                right_eye = [(int(face_landmarks.landmark[i].x * w),
                              int(face_landmarks.landmark[i].y * h)) for i in self.RIGHT_EYE_IDX]

                # Hitung EAR kiri & kanan
                left_ear = self._eye_aspect_ratio(left_eye)
                right_ear = self._eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Cek blink
                if ear < self.EAR_THRESHOLD:
                    self.counter += 1
                else:
                    if self.counter >= self.CONSEC_FRAMES:
                        self.total_blinks += 1
                        blink_detected = True
                    self.counter = 0

                # Gambar mata & info EAR
                # cv2.polylines(frame, [np.array(left_eye)], True, (50, 255, 50), 1)
                # cv2.polylines(frame, [np.array(right_eye)], True, (50, 255, 50), 1)
                cv2.putText(frame, f"EAR: {ear:.2f}", (15, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1)
                cv2.putText(frame, f"Blinks: {self.total_blinks}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1)

        return frame, blink_detected, self.total_blinks
