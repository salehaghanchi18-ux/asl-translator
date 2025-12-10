from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import torch
import joblib
import numpy as np
import torch.nn.functional as F
from collections import deque
from train_alphabet_model import GestureNet
import pyttsx3

app = Flask(__name__)

# Load model + encoder + scaler
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("feature_scaler.pkl")
mean = scaler["mean"].astype(np.float32)
std = scaler["std"].astype(np.float32)

model = GestureNet(63, len(label_encoder.classes_))
model.load_state_dict(torch.load("asl_alphabet_model.pt"))
model.eval()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# State
window = deque(maxlen=7)
CONF_THRESH = 0.60
mirror = True
text_buffer = ""
latest_status = {"label": "-", "conf": 0.0, "top2": ["-", 0.0], "margin": 0.0, "hand": False}

# TTS
tts_engine = pyttsx3.init()

def scale_features(vec):
    return (vec - mean) / std

def majority_vote(q):
    if not q: return "?"
    vals, counts = np.unique(q, return_counts=True)
    return vals[np.argmax(counts)]

def classify(vec63):
    vec_scaled = scale_features(vec63)
    x = torch.tensor([vec_scaled], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx_sorted = np.argsort(-probs)
    cls_idx = int(idx_sorted[0])
    conf = float(probs[cls_idx])
    top2_idx = int(idx_sorted[1]) if probs.shape[0] > 1 else cls_idx
    top2 = [label_encoder.inverse_transform([top2_idx])[0], float(probs[top2_idx])]
    letter = label_encoder.inverse_transform([cls_idx])[0]
    margin = conf - top2[1]
    return letter, conf, top2, margin

def gen_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 280)

    while True:
        ok, frame = cap.read()
        if not ok:
            latest_status.update({"label": "-", "conf": 0.0, "top2": ["-", 0.0], "margin": 0.0, "hand": False})
            err = np.zeros((280, 640, 3), dtype=np.uint8)
            cv2.putText(err, "Camera/Network issue...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            ret, jpg = cv2.imencode('.jpg', err)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
            continue

        if mirror:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        pred_text, color = "No hand detected", (0, 0, 255)
        latest_status["hand"] = False

        if result.multi_hand_landmarks:
            latest_status["hand"] = True
            hl = result.multi_hand_landmarks[0]
            coords = []
            for lm in hl.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            letter, conf, top2, margin = classify(np.array(coords, dtype=np.float32))
            window.append(letter)
            smooth_letter = majority_vote(window)
            latest_status.update({"label": smooth_letter, "conf": conf, "top2": top2, "margin": margin})

            if conf >= CONF_THRESH:
                pred_text = f"Prediction: {smooth_letter} ({conf:.2f})"
                color = (0, 200, 0)
            else:
                pred_text = f"Low confidence → {smooth_letter} ({conf:.2f})"
                color = (0, 165, 255)

            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, pred_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (8, 48), (632, 232), (255, 105, 180), 2)

        ret, jpg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/translate')
def translate():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    return jsonify({
        "label": latest_status["label"],
        "conf": round(latest_status["conf"], 2),
        "top2": [latest_status["top2"][0], round(latest_status["top2"][1], 2)],
        "margin": round(latest_status["margin"], 2),
        "hand": latest_status["hand"],
        "text": text_buffer
    })

@app.route('/api/confirm', methods=['POST'])
def confirm():
    global text_buffer
    letter = latest_status["label"]
    if letter and letter != "?":
        text_buffer += letter
    return jsonify({"text": text_buffer})

@app.route('/api/backspace', methods=['POST'])
def backspace():
    global text_buffer
    text_buffer = text_buffer[:-1] if text_buffer else ""
    return jsonify({"text": text_buffer})

@app.route('/api/clear', methods=['POST'])
def clear():
    global text_buffer
    text_buffer = ""
    return jsonify({"text": text_buffer})

@app.route('/api/speak', methods=['POST'])
def speak():
    text = request.json.get("text", "")
    if text:
        tts_engine.say(text)
        tts_engine.runAndWait()
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(debug=True)
