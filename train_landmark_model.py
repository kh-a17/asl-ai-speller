import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. LOAD AND NORMALIZE DATA
df = pd.read_csv("asl_landmarks.csv")

# Extract features and labels
X_raw = df.drop("label", axis=1).values
y_raw = df["label"].values

def normalize_landmarks(data):
    """Normalization for Training (Input is a NumPy array from CSV)"""
    normalized_data = []
    for row in data:
        # 1. Reshape the flat 42 numbers back to (21, 2)
        coords = row.reshape(21, 2)
        
        # 2. Make relative to wrist (Point 0)
        wrist = coords[0]
        coords = coords - wrist
        
        # 3. Scale normalize (divide by the distance of the furthest point)
        max_distance = np.max(np.linalg.norm(coords, axis=1))
        if max_distance > 0:
            coords = coords / max_distance
            
        normalized_data.append(coords.flatten())
    return np.array(normalized_data)

# Now this will work with your CSV data:
X_norm = normalize_landmarks(X_raw)

X_norm = normalize_landmarks(X_raw)

# 2. ENCODE LABELS
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_raw)
num_classes = len(encoder.classes_)

# 3. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y_encoded, test_size=0.2, random_state=42
)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 4. MODEL ARCHITECTURE
class ASLModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),  # Normalizes activations to speed up training
            nn.ReLU(),
            nn.Dropout(0.2),      # Prevents overfitting by randomly silencing neurons
            
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

model = ASLModel(42, num_classes)

# 5. OPTIMIZER & LOSS
criterion = nn.CrossEntropyLoss()
# Lower learning rate (0.001) for smoother learning
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. TRAINING LOOP
epochs = 100 # Increased epochs
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Calculate Accuracy
    if (epoch + 1) % 2 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, predicted = torch.max(test_outputs, 1)
            acc = (predicted == y_test).float().mean()
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Test Acc: {acc*100:.2f}%")

# 7. SAVE MODEL & METADATA
torch.save({
    "model_state": model.state_dict(),
    "labels": encoder.classes_
}, "asl_landmark_model.pth")

print("\nModel saved successfully as asl_landmark_model.pth!")