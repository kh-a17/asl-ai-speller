import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

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

# ===== MEDIAPIPE SETUP =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===== NORMALIZATION FUNCTION =====
def normalize_landmarks(landmarks):
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    
    # Make relative to wrist
    wrist = coords[0]
    coords = coords - wrist
    
    # Scale normalize (divide by max distance)
    max_distance = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / max_distance
    
    return coords.flatten()


# ===== WEBCAM =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract + normalize
        input_data = normalize_landmarks(hand_landmarks.landmark)
        input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction_text = labels[predicted.item()]

    # Display prediction
    cv2.rectangle(frame, (0, 0), (300, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Prediction: {prediction_text}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("ASL Landmark Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
