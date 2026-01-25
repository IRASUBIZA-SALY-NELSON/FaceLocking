import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
from config import MODEL_PATH

# 5-point template (standard ArcFace 112x112)
REF_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041]  # right mouth
], dtype=np.float64)

# Corresponding MediaPipe indices (with frame flip for mirror view)
LM_INDICES = [33, 263, 1, 61, 291]  # left eye, right eye, nose, left mouth, right mouth

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

session = ort.InferenceSession(MODEL_PATH)


def calculate_similarity_transform(src_pts, dst_pts):
    src_pts = np.array(src_pts, dtype=np.float64)
    dst_pts = np.array(dst_pts, dtype=np.float64)

    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)

    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    var_src = np.mean(np.linalg.norm(src_centered, axis=1) ** 2)
    scale = np.sum(S) / var_src if var_src > 0 else 1.0

    t = dst_mean - scale * R @ src_mean

    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


def get_landmarks(rgb_frame):
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None


def get_5_points(landmarks, h, w):
    return np.array([
        [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in LM_INDICES
    ], dtype=np.float64)


def align_face(face_crop, landmarks_pts):
    M = calculate_similarity_transform(landmarks_pts, REF_POINTS)
    aligned = cv2.warpAffine(face_crop, M[:2], (112, 112), flags=cv2.INTER_LINEAR)
    return aligned


def preprocess(aligned):
    img = aligned.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def get_embedding(aligned):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    blob = preprocess(aligned)
    emb = session.run([output_name], {input_name: blob})[0][0]
    emb = emb / np.linalg.norm(emb)
    return emb