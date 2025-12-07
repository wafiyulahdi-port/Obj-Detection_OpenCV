import cv2
import numpy as np
import onnxruntime as ort
import os
import json

class ArcFaceEmbedder:
    def __init__(self, model_root: str = "models", model_name: str = "arcface_r100_v1", providers=None):

        if providers is None:
            providers = ["CPUExecutionProvider"]

        model_path = os.path.join(model_root, model_name, "arcface.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")

        # Load model ONNX
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        config_path = os.path.join(model_root, model_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {"image_size": [112, 112]}  # Default

    def preprocess(self, face_aligned: np.ndarray) -> np.ndarray:

        size = tuple(self.config.get("image_size", [112, 112]))
        face_rgb = cv2.cvtColor(face_aligned, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, size)
        face_rgb = (face_rgb / 127.5 - 1.0).astype(np.float32)  # Normalisasi ke [-1,1]
        face_rgb = np.transpose(face_rgb, (2, 0, 1))  # HWC -> CHW
        face_rgb = np.expand_dims(face_rgb, axis=0)   #  Batch
        return np.ascontiguousarray(face_rgb)

    def embed(self, face_aligned: np.ndarray) -> np.ndarray:
        img = self.preprocess(face_aligned)
        embedding = self.session.run(None, {self.input_name: img})[0]
        # Normalisasi L2
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.flatten()
