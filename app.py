from flask import Flask, render_template, jsonify, request, Response
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
from collections import deque
import time

app = Flask(__name__)

# ========== LOAD YOUR EXISTING MODEL ==========
try:
    with open("sign_model.pkl", "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    le = data["le"]
    print("✅ Model loaded successfully!")
    print("Classes:", list(le.classes_))
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    le = None

# ========== GLOBAL VARIABLES ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Webcam and processing variables
cap = None
is_camera_active = False
is_processing = False
latest_frame = None
frame_lock = threading.Lock()

# Prediction stability
prediction_buffer = deque(maxlen=10)
current_prediction = "Show your sign..."
prev_spoken = None
last_spoken_time = 0
cooldown = 1.5

# ========== TEXT-TO-SPEECH ==========
def speak(text):
    def _inner():
        try:
            e = pyttsx3.init()
            e.setProperty('rate', 150)
            e.say(text)
            e.runAndWait()
        except Exception as e:
            print(f"❌ TTS Error: {e}")
    threading.Thread(target=_inner, daemon=True).start()

# ========== SIMPLE ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, is_camera_active, is_processing
    
    if not is_camera_active:
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"status": "error", "message": "Cannot access camera"})
            
            is_camera_active = True
            is_processing = True
            
            # Start processing thread
            threading.Thread(target=process_camera_feed, daemon=True).start()
            print("📹 Camera started and processing...")
            return jsonify({"status": "camera_started"})
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return jsonify({"status": "error", "message": str(e)})
    
    return jsonify({"status": "already_running"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, is_camera_active, is_processing
    
    if is_camera_active and cap is not None:
        is_processing = False
        time.sleep(0.5)  # Let the thread finish
        
        cap.release()
        cv2.destroyAllWindows()
        is_camera_active = False
        
        # Reset prediction
        global current_prediction
        current_prediction = "Show your sign..."
        
        print("📹 Camera stopped")
        return jsonify({"status": "camera_stopped"})
    
    return jsonify({"status": "already_stopped"})

@app.route('/get_prediction')
def get_prediction():
    global current_prediction
    return jsonify({"prediction": current_prediction})

@app.route('/speak_text', methods=['POST'])
def speak_text():
    text = request.json.get('text', '')
    if text and text != "Show your sign..." and text != "Speech will appear here...":
        speak(text)
        return jsonify({"status": "speaking", "text": text})
    return jsonify({"status": "no_text"})

# ========== VIDEO STREAMING ==========
def generate_frames():
    global latest_frame, is_processing
    while is_processing:
        with frame_lock:
            if latest_frame is not None:
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame = buffer.tobytes()
                
                # Yield frame in HTTP response format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ========== CAMERA PROCESSING ==========
def process_camera_feed():
    global cap, current_prediction, prediction_buffer, prev_spoken, last_spoken_time, is_processing, latest_frame
    
    frame_count = 0
    print("🎥 Starting camera processing...")
    
    while is_processing and is_camera_active and cap is not None:
        try:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from camera")
                break

            frame_count += 1
            
            # Store latest frame for streaming
            with frame_lock:
                latest_frame = frame.copy()
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Skip some frames for performance (process every 2nd frame)
            if frame_count % 2 != 0:
                continue

            # Process with MediaPipe
            res = hands.process(img_rgb)
            temp_prediction = None

            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    # Draw landmarks on frame
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    row = []
                    for lm in handLms.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    X = np.array(row).reshape(1, -1)

                    # Ensure correct feature length
                    if X.shape[1] != model.n_features_in_:
                        diff = model.n_features_in_ - X.shape[1]
                        X = np.hstack([X, np.zeros((1, diff))])

                    # Make prediction
                    pred_enc = model.predict(X)[0]
                    pred_label = le.inverse_transform([pred_enc])[0]
                    temp_prediction = pred_label
                    
                    # Display prediction on frame
                    cv2.putText(frame, f"Pred: {pred_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    print(f"🤖 Detected: {pred_label}")

            # ========== STABILITY FILTER ==========
            if temp_prediction:
                prediction_buffer.append(temp_prediction)
                print(f"📊 Buffer: {list(prediction_buffer)}")

                if len(prediction_buffer) == prediction_buffer.maxlen and len(set(prediction_buffer)) == 1:
                    stable_prediction = prediction_buffer[0]
                    current_prediction = stable_prediction

                    # Speak only if changed and cooldown passed
                    current_time = time.time()
                    if stable_prediction != prev_spoken and (current_time - last_spoken_time) > cooldown:
                        print(f"🔊 Speaking: {stable_prediction}")
                        speak(stable_prediction)
                        prev_spoken = stable_prediction
                        last_spoken_time = current_time
            else:
                # No hand detected
                if current_prediction != "Show your sign...":
                    current_prediction = "Show your sign..."
                    print("👋 No hand detected")

            # Update the frame with drawings
            with frame_lock:
                latest_frame = frame

            # Small delay to prevent overwhelming CPU
            time.sleep(0.05)
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            break
    
    print("🎥 Camera processing stopped")

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)