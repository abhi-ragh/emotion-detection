import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Real-time Emotion Detection')
parser.add_argument('--model', type=str, default='emotion_model_final.keras', 
                    help='Path to the trained emotion model')
parser.add_argument('--haar_cascade', type=str, default='haarcascade_frontalface_default.xml',
                    help='Path to the Haar cascade file')
parser.add_argument('--camera', type=int, default=0,
                    help='Camera device index')
parser.add_argument('--confidence', type=float, default=0.5,
                    help='Minimum probability to filter weak detections')
parser.add_argument('--show_fps', type=bool, default=True,
                    help='Display FPS counter')
args = parser.parse_args()

# Define emotion labels
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Load the emotion detection model
print("[INFO] Loading emotion detection model...")
try:
    if args.model.endswith('.keras'):
        model = load_model(args.model)
    else:
        model = load_model(args.model, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("[INFO] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Load the face detector
print("[INFO] Loading face detector...")
try:
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + args.haar_cascade)
    if face_detector.empty():
        raise Exception("Haar cascade file not found or invalid")
    print("[INFO] Face detector loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load face detector: {e}")
    exit(1)

# Function to preprocess the detected face for emotion classification
def preprocess_face(face):
    # Convert to grayscale if it's not already
    if len(face.shape) == 3 and face.shape[2] == 3:
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    else:
        face_gray = face
    
    # Resize to match the input size expected by the model (48x48)
    face_resized = cv2.resize(face_gray, (48, 48))
    
    # Normalize pixel values
    face_normalized = face_resized / 255.0
    
    # Reshape to match model input shape [batch_size, height, width, channels]
    face_input = np.expand_dims(np.expand_dims(face_normalized, -1), 0)
    
    return face_input

# Function to get emotion prediction
def predict_emotion(face):
    # Preprocess the face
    face_input = preprocess_face(face)
    
    # Get model prediction
    predictions = model.predict(face_input, verbose=0)
    
    # Get the predicted emotion label and confidence
    emotion_idx = np.argmax(predictions[0])
    emotion_label = emotions[emotion_idx]
    confidence = predictions[0][emotion_idx] * 100
    
    return emotion_label, confidence, predictions[0]

# Function to draw emotion bars
def draw_emotion_bars(frame, predictions, x, y, w, h):
    bar_width = 100
    bar_height = 10
    spacing = 15
    max_bar_height = len(emotions) * (bar_height + spacing)
    
    # Draw background for emotion bars
    cv2.rectangle(frame, 
                  (x + w + 10, y), 
                  (x + w + 10 + bar_width + 50, y + max_bar_height), 
                  (20, 20, 20), 
                  -1)
    
    # Draw bars for each emotion
    for i, emotion in enumerate(emotions):
        confidence = predictions[i] * 100
        # Determine bar color (green for high confidence, blend to red for low)
        r = int(255 * (1 - confidence/100))
        g = int(255 * (confidence/100))
        b = 0
        color = (b, g, r)
        
        # Draw filled rectangle representing the confidence
        bar_length = int(confidence / 100 * bar_width)
        cv2.rectangle(frame, 
                      (x + w + 10, y + i * spacing), 
                      (x + w + 10 + bar_length, y + i * spacing + bar_height), 
                      color, 
                      -1)
        
        # Draw emotion label with percentage
        cv2.putText(frame, 
                    f"{emotion}: {confidence:.1f}%", 
                    (x + w + 15 + bar_width, y + i * spacing + bar_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1)

# Initialize webcam
print("[INFO] Starting video stream...")
video_capture = cv2.VideoCapture(args.camera)

# Check if camera opened successfully
if not video_capture.isOpened():
    print("[ERROR] Could not open webcam")
    exit(1)

# Set video properties (optional)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Variables for FPS calculation
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

print("[INFO] Press 'q' to quit")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    
    if not ret:
        print("[ERROR] Failed to grab frame")
        break
    
    # Create a copy for display
    display_frame = frame.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        
        # Predict emotion
        emotion_label, confidence, all_predictions = predict_emotion(face_roi)
        
        # Draw rectangle around the face
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display emotion label
        label_text = f"{emotion_label}: {confidence:.1f}%"
        cv2.putText(display_frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw emotion confidence bars
        draw_emotion_bars(display_frame, all_predictions, x, y, w, h)
    
    
    # Display instructions
    cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', display_frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Application terminated")