import cv2
import mediapipe as mp
import pandas as pd
import os

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Ask user for label
label = input("Enter gesture label (e.g. A, B, C, D): ").strip().upper()
print(f"Recording gesture: {label} | Press 'q' to stop")

data = []
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = []
            # Collect 21 landmarks (x,y,z → 63 values)
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            # Append label at the end
            coords.append(label)
            data.append(coords)
            count += 1

    # Show recording info
    cv2.putText(frame, f"Label: {label} | Samples: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.imshow("Gesture Recorder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save CSV
if not os.path.exists("gesture_data"):
    os.makedirs("gesture_data")

if data:
    df = pd.DataFrame(data)
    # No header, just raw values
    df.to_csv(f"gesture_data/gesture_data_{label}.csv", index=False, header=False)
    print(f"✅ Saved {count} samples to gesture_data_{label}.csv")
else:
    print("⚠️ No data recorded.")
