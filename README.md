# Face Recognition with ArcFace ONNX and 5-Point Alignment

Reproduction of the book by Gabriel Baziramwabo.

## Setup
- Install dependencies: `pip install -r requirements.txt`
- Download model into `models/embedder_arcface.onnx` (see instructions above)

## Validation Tests (Chapter 1)
Run each: `python src/camera_test.py`, `python src/face_detection_test.py`, etc.

## Enrollment
`python src/enroll.py` → Enroll ≥10 identities with 15+ samples each.

## Threshold Evaluation
`python src/threshold_evaluation.py` → Observe distributions and update THRESHOLD in `src/config.py`.

## Live Recognition
`python src/live_recognition.py`

## How Enrollment Works
Multiple samples per identity → stored as list of L2-normalized embeddings → mean used for comparison.

## Threshold Used
0.62.