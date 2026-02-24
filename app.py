from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# ===== LOAD MODEL =====
checkpoint = torch.load("asl_landmark_model.pth", map_location="cpu")
labels = checkpoint["labels"]

class ASLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.model(x)

model = ASLModel(42, len(labels))
model.load_state_dict(checkpoint["model_state"])
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # Make relative to wrist
    wrist = coords[0]
    coords = coords - wrist
    
    # Scale normalize (divide by max distance)
    max_distance = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / max_distance
    
    return coords.flatten()


cap = cv2.VideoCapture(0)
latest_letter = ""  # Global variable

def generate_frames():
    global latest_letter

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            input_data = normalize_landmarks(landmarks.landmark)
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                conf, predicted = torch.max(probs, 1)

                if conf.item() > 0.55:
                    latest_letter = labels[predicted.item()]
                else:
                    latest_letter = ""
        else:
            latest_letter = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letter')
def get_letter():
    return jsonify({"letter": latest_letter})

if __name__ == "__main__":
    app.run(debug=True)