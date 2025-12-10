import cv2
import mediapipe as mp
import torch
import joblib
import numpy as np
from collections import deque
import torch.nn.functional as F

# Import same architecture
from train_alphabet_model import GestureNet

# Load encoder, scaler, and model
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("feature_scaler.pkl")  # dict with mean, std
mean = scaler["mean"].astype(np.float32)
std = scaler["std"].astype(np.float32)

input_size = 63
num_classes = len(label_encoder.classes_)
model = GestureNet(input_size, num_classes)
model.load_state_dict(torch.load("asl_alphabet_model.pt"))
model.eval()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("📷 Camera started. Press 'q' to quit.")

# Smoothing
window = deque(maxlen=7)
CONF_THRESH = 0.60

def majority_vote(q):
    if not q:
        return "?"
    vals, counts = np.unique(q, return_counts=True)
    return vals[np.argmax(counts)]

def scale_features(vec63):
    # vec63: np.array shape (63,)
    return (vec63 - mean) / std

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pred_text, color = "No hand detected", (0, 0, 255)

    if result.multi_hand_landmarks:
        hl = result.multi_hand_landmarks[0]
        coords = []
        for lm in hl.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        vec = np.array(coords, dtype=np.float32)

        # Scale with training scaler
        vec_scaled = scale_features(vec)

        X = torch.tensor([vec_scaled], dtype=torch.float32)
        with torch.no_grad():
            logits = model(X)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            cls_idx = int(np.argmax(probs))
            conf = float(probs[cls_idx])
            letter = label_encoder.inverse_transform([cls_idx])[0]

        window.append(letter)
        smooth_letter = majority_vote(window)

        if conf >= CONF_THRESH:
            pred_text = f"Prediction: {smooth_letter} ({conf:.2f})"
            color = (0, 200, 0)
        else:
            pred_text = f"Low confidence → {smooth_letter} ({conf:.2f})"
            color = (0, 165, 255)

        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, pred_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("ASL Translator", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
