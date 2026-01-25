import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "embedder_arcface.onnx")
DB_PATH = os.path.join(BASE_DIR, "data", "db", "face_db.pkl")
INPUT_SIZE = (112, 112)
THRESHOLD = 0.62