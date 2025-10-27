import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
from collections import deque
import time

# ========== Load model + encoder ==========
with open("sign_model.pkl", "rb") as f:
    data = pickle.load(f)
model = data["model"]
le = data["le"]

print("Classes:", list(le.classes_))
print("📏 Model expects", model.n_features_in_, "features per sample")

# ========== Text-to-Speech (fresh engine per call) ==========
def speak(text):
    def _inner():
        e = pyttsx3.init()
        e.setProperty('rate', 150)
        e.say(text)
        e.runAndWait()
    threading.Thread(target=_inner, daemon=True).start()

# ========== MediaPipe setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ========== Open webcam ==========
cap = cv2.VideoCapture(0)

# Stability buffer (longer = more stable)
prediction_buffer = deque(maxlen=10)
prev_spoken = None
frame_count = 0
last_spoken_time = 0
cooldown = 1.5  # seconds between speeches

print("\n✋ Show your sign to the camera... Press ESC to quit.")

# ========== Main loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Skip frames for speed (only process every 3rd frame)
    if frame_count % 3 != 0:
        cv2.imshow("Sign Language Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    res = hands.process(img_rgb)
    current_prediction = None

    if res.multi_hand_landmarks:
        for handLms in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            row = []
            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

            X = np.array(row).reshape(1, -1)

            # Ensure correct feature length
            if X.shape[1] != model.n_features_in_:
                diff = model.n_features_in_ - X.shape[1]
                X = np.hstack([X, np.zeros((1, diff))])

            pred_enc = model.predict(X)[0]
            pred_label = le.inverse_transform([pred_enc])[0]
            current_prediction = pred_label

            cv2.putText(frame, f"Prediction: {pred_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # ========== Stability filter ==========
    if current_prediction:
        prediction_buffer.append(current_prediction)

        # If last 10 predictions are same → stable sign
        if len(prediction_buffer) == prediction_buffer.maxlen and len(set(prediction_buffer)) == 1:
            stable_prediction = prediction_buffer[0]

            # Speak only if changed and cooldown passed
            if stable_prediction != prev_spoken and (time.time() - last_spoken_time) > cooldown:
                speak(stable_prediction)
                prev_spoken = stable_prediction
                last_spoken_time = time.time()

    cv2.imshow("Sign Language Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
