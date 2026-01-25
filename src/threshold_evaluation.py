import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import DB_PATH

with open(DB_PATH, 'rb') as f:
    db = pickle.load(f)

genuine_scores = []
impostor_scores = []

# Genuine pairs
for name, embs in db.items():
    embs = np.array(embs)
    if len(embs) < 2:
        continue
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = np.dot(embs[i], embs[j])
            genuine_scores.append(sim)

# Impostor pairs (sample some to avoid too many)
for name1, embs1 in db.items():
    for name2, embs2 in db.items():
        if name1 >= name2:
            continue
        embs1_arr = np.array(embs1)
        embs2_arr = np.array(embs2)
        for e1 in embs1_arr:
            for e2 in embs2_arr[:10]:  # Limit for speed
                sim = np.dot(e1, e2)
                impostor_scores.append(sim)

genuine_scores = np.array(genuine_scores)
impostor_scores = np.array(impostor_scores)

print(f"Genuine mean: {genuine_scores.mean():.3f} ± {genuine_scores.std():.3f}")
print(f"Impostor mean: {impostor_scores.mean():.3f} ± {impostor_scores.std():.3f}")

plt.hist(genuine_scores, bins=50, alpha=0.7, label='Genuine')
plt.hist(impostor_scores, bins=50, alpha=0.7, label='Impostor')
plt.legend()
plt.xlabel('Cosine Similarity')
plt.title('Threshold Evaluation')
plt.axvline(0.50, color='red', linestyle='--', label='Suggested Threshold ~0.50')
plt.show()

print("Update THRESHOLD in config.py based on the separation (e.g., 0.45-0.55).")