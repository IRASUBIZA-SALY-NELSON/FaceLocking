import cv2
import numpy as np
from utils import detector, get_landmarks, get_5_points, align_face, get_embedding

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
        crop = frame[y:y+h, x:x+w]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(rgb)
        if landmarks:
            pts = get_5_points(landmarks, h, w)
            aligned = align_face(crop, pts)
            emb = get_embedding(aligned)
            print("Embedding shape:", emb.shape, "| Norm:", np.linalg.norm(emb))
            cv2.imshow('Aligned', aligned)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()