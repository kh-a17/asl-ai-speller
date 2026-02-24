import os
import cv2
import mediapipe as mp
import pandas as pd

# ===== CONFIG =====
dataset_path = "dataset/asl_alphabet_train"
output_csv = "asl_landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

data = []

# Loop through each letter folder
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing letter: {label}")

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            row = []

            for lm in landmarks.landmark:
                row.append(lm.x)
                row.append(lm.y)

            row.append(label)
            data.append(row)

# Create DataFrame
columns = []
for i in range(21):
    columns += [f"x{i}", f"y{i}"]
columns.append("label")

df = pd.DataFrame(data, columns=columns)
df.to_csv(output_csv, index=False)

print("Landmarks saved to asl_landmarks.csv")
