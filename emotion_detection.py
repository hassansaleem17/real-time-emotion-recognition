# =====================================
# Real-Time Face Emotion Recognition (macOS version)
# =====================================

# Install required packages (uncomment the line below if needed)
# !pip install opencv-python opencv-contrib-python deepface

import cv2
from deepface import DeepFace

# Load Haar Cascade for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try to open available camera indexes automatically (macOS fix)
cap = None
for i in range(3):
    temp_cap = cv2.VideoCapture(i)
    if temp_cap.isOpened():
        cap = temp_cap
        print(f"✅ Webcam opened successfully at index {i}")
        break

if cap is None or not cap.isOpened():
    raise IOError("❌ Cannot open webcam. Please check camera permissions in System Settings → Privacy & Security → Camera.")

# Start video capture loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured properly.")
        break

    # Analyze emotions using DeepFace
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        print("Error analyzing frame:", e)
        continue

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract dominant emotion
    dominant_emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']

    # Display emotion text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)

    # Show video feed
    cv2.imshow('Real-Time Emotion Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
