**ASL AI Word Builder**
Real-time American Sign Language (ASL) Alphabet Recognition with Speech Output

ASL AI Word Builder is a real-time computer vision web application that recognizes American Sign Language (ASL) alphabet gestures from a webcam feed and converts them into text and speech. The system uses hand landmark detection and a neural network classifier to predict ASL letters live, allowing users to build words interactively and hear them spoken aloud.


**Demo**

https://github.com/user-attachments/assets/0b656429-d09a-463e-a011-0b50f5ade154


**How It Works**
- Webcam captures live video frames
- Hand landmarks are extracted using MediaPipe
- Landmarks are normalized (wrist-relative + scale normalization)
- A trained PyTorch neural network predicts the ASL letter
- The Flask backend streams predictions to the frontend
- The user builds words and triggers speech output using the browser's Speech API


**Architecture**

Webcam → MediaPipe → Landmark Normalization → PyTorch Model → Flask Backend → Frontend UI → Speech Synthesis


**Dataset**
The model was trained using the **ASL Alphabet Dataset** available on Kaggle.

- **Source:** [Kaggle - ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Dataset Structure:** 87,000 images of 200x200 pixels.
- **Classes:** 29 total (Letters A-Z, plus space, delete, and nothing).
- **Processing:** For this project, only the **landmarks** (coordinates) were used for training to ensure the model remains lightweight and focused on hand geometry rather than background pixels.


**How to run locally**

- git clone [https://github.com/YOUR_USERNAME/asl-ai-speller.git](https://github.com/kh-a17/asl-ai-speller.git)

- cd asl-ai-speller

- python -m venv venv

  - source venv/bin/activate   # Mac/Linux
  - venv\Scripts\activate      # Windows

- pip install -r requirements.txt

- python app.py
