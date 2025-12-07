from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional


def _align_face(image_bgr: np.ndarray,
                keypoints: Dict[str, Tuple[int, int]],
                output_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
    """Align wajah menggunakan 2 mata + hidung (affine)."""
    src_pts = np.float32([
        keypoints['left_eye'],
        keypoints['right_eye'],
        keypoints['nose']
    ])

    W, H = output_size
    tgt_pts = np.float32([
        (0.30 * W, 0.40 * H),  # left_eye
        (0.70 * W, 0.40 * H),  # right_eye
        (0.50 * W, 0.60 * H)   # nose
    ])


    M = cv2.getAffineTransform(src_pts, tgt_pts)
    aligned = cv2.warpAffine(image_bgr, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT101)
    return aligned


class FaceDetector:
    def __init__(
        self,
        device: Optional[str] = None,
        do_flip = True,
        threshold: float = 0.90,
        keep_all: bool = True,
        downscale: float = 0.75,
        mirror_preview: bool = False 
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.do_flip = do_flip
        self.threshold = threshold
        self.keep_all = keep_all
        self.downscale = float(downscale)
        self.mirror_preview = mirror_preview

        self.detector = MTCNN(keep_all=True, device=self.device)

    def _prep_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Hanya downscale (tanpa flip)."""
        img = frame_bgr
        scale = 1.0
        if self.downscale and self.downscale != 1.0:
            img = cv2.resize(img, (0, 0), fx=self.downscale, fy=self.downscale,
                             interpolation=cv2.INTER_LINEAR)
            scale = self.downscale
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb, scale

    def detect_faces(self, frame_bgr: np.ndarray) -> List[Dict]:
        """Deteksi wajah + alignment."""
        rgb_img, scale = self._prep_frame(frame_bgr)

        boxes, probs, landmarks = self.detector.detect(rgb_img, landmarks=True)
        faces = []

        if boxes is not None:
            inv = 1.0 / scale
            for box, prob, kps in zip(boxes, probs, landmarks):
                if prob is None or prob < self.threshold:
                    continue

                x1, y1, x2, y2 = [int(round(b * inv)) for b in box]

                names = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
                kdict = {name: (int(round(pt[0] * inv)), int(round(pt[1] * inv)))
                         for name, pt in zip(names, kps)}

                # clamp
                H, W = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)

                face = {
                    'box': (x1, y1, x2, y2),
                    'confidence': float(prob),
                    'keypoints': kdict
                }

                try:
                    face['aligned'] = _align_face(frame_bgr, kdict, (160, 160))
                except Exception:
                    face['aligned'] = cv2.resize(frame_bgr[y1:y2, x1:x2], (160, 160)) \
                        if (x2 > x1 and y2 > y1) else None

                faces.append(face)

        if not faces:
            return []

        return faces if self.keep_all else [max(faces, key=lambda f: f['confidence'])]

    def draw_faces(self, image_bgr: np.ndarray, faces: List[Dict]) -> np.ndarray:
        img = image_bgr.copy()

        for face in faces:
            x1, y1, x2, y2 = face['box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (50, 255, 50), 2)

            for point in face['keypoints'].values():
                cv2.circle(img, point, 2, (0, 0, 255), -1)

        if self.mirror_preview:
            img = cv2.flip(img, 1)

        return img
