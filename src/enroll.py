import cv2
import pickle
import os
from utils import detector, get_landmarks, get_5_points, align_face, get_embedding
from config import DB_PATH

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

if os.path.exists(DB_PATH):
    with open(DB_PATH, 'rb') as f:
        db = pickle.load(f)
else:
    db = {}  # name -> list of embeddings

name = input("Enter identity name: ").strip()
if name in db:
    print(f"{name} already exists. Adding more samples.")

embeddings = []

cap = cv2.VideoCapture(0)
count = 0
required = 15  # Recommended multiple samples

print("Press SPACE to capture sample, Q to finish")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(rgb)
        if landmarks:
            pts = get_5_points(landmarks, h, w)
            aligned = align_face(crop, pts)
            emb = get_embedding(aligned)
            embeddings.append(emb)
            count += 1
            cv2.putText(frame, f"Captured: {count}/{required}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.imshow('Aligned Sample', aligned)
    cv2.imshow('Enroll - Press SPACE', frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        pass  # Capture happens automatically on detection
    if key == ord('q'):
        break

if embeddings:
    if name not in db:
        db[name] = []
    db[name].extend(embeddings)
    with open(DB_PATH, 'wb') as f:
        pickle.dump(db, f)
    print(f"Enrolled {len(embeddings)} samples for {name}. Total: {len(db[name])}")
else:
    print("No samples captured.")

cap.release()
cv2.destroyAllWindows()