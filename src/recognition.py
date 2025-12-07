import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embedding import ArcFaceEmbedder

class FaceRecognizer:
    def __init__(self, threshold: float = 0.80, db_path: str = "output/db_faces.npz", data_path: str = "data/employees"):
        self.db_embeddings = []
        self.db_names = []
        self.threshold = threshold
        self.db_path = db_path
        self.data_path = data_path

    def add_identity(self, name: str, embedding: np.ndarray):
        self.db_names.append(name)
        self.db_embeddings.append(embedding)

    def recognize(self, embedding: np.ndarray):
        if not self.db_embeddings:
            return "Unknown", 0.0
        sims = cosine_similarity([embedding], self.db_embeddings)[0]
        best_idx = np.argmax(sims)
        if sims[best_idx] >= self.threshold:
            return self.db_names[best_idx], sims[best_idx]
        else:
            return "Unknown", sims[best_idx]

    def save_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        np.savez(self.db_path, embeddings=self.db_embeddings, names=self.db_names)
        print(f"[Database disimpan ke {self.db_path}")

    def load_database(self):
        if os.path.exists(self.db_path):
            data = np.load(self.db_path, allow_pickle=True)
            self.db_embeddings = list(data["embeddings"])
            self.db_names = list(data["names"])
            print(f"Database dimuat dari {self.db_path} ({len(self.db_names)} orang)")
        else:
            print("Database kosong, silakan tambahkan wajah dengan 's'")

    def build_from_images(self, embedder: ArcFaceEmbedder):
        """
        Auto load semua gambar dari data/employees/<nama>/*.jpg
        """
        if not os.path.exists(self.data_path):
            return
        for person_name in os.listdir(self.data_path):
            person_dir = os.path.join(self.data_path, person_name)
            if not os.path.isdir(person_dir):
                continue
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(person_dir, img_file)
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    # resize kalau terlalu besar
                    img = cv2.resize(img, (160, 160))
                    emb = embedder.embed(img)
                    self.add_identity(person_name, emb)
        print(f"Loaded {len(self.db_names)} identities dari {self.data_path}")
