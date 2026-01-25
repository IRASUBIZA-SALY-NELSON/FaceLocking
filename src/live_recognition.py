import cv2
import numpy as np
import pickle
from utils import detector, get_landmarks, get_5_points, align_face, get_embedding
from config import DB_PATH, THRESHOLD

with open(DB_PATH, 'rb') as f:
    db = pickle.load(f)

# Precompute mean embeddings
reference = {}
for name, embs in db.items():
    mean_emb = np.mean(embs, axis=0)
    mean_emb /= np.linalg.norm(mean_emb)
    reference[name] = mean_emb

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop = frame[y:y + h, x:x + w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(rgb)
        if landmarks:
            pts = get_5_points(landmarks, h, w)
            aligned = align_face(crop, pts)
            query_emb = get_embedding(aligned)

            max_sim = -1
            identity = "Unknown"
            for name, ref_emb in reference.items():
                sim = np.dot(query_emb, ref_emb)
                if sim > max_sim:
                    max_sim = sim
                    identity = name

            label = identity if max_sim >= THRESHOLD else "Unknown"
            text = f"{label} ({max_sim:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if label != "Unknown" else (0, 0, 255), 2)
            cv2.imshow('Aligned', aligned)
    cv2.imshow('Live Recognition', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()