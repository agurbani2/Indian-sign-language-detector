import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
import numpy as np
import time

# Load the trained model and label encoder
model = tf.keras.models.load_model('/Users/apple/Desktop/isl/word_model.keras')

with open('/Users/apple/Desktop/isl/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Timer
start_time = time.time()
wait_time = 7  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)

    elapsed_time = time.time() - start_time
    if elapsed_time >= wait_time and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Flatten landmarks into shape (63,)
            flattened_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Predict using the model
            prediction = model.predict(np.expand_dims(flattened_landmarks, axis=0))
            predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))[0]

            print(f"🧠 Predicted Sign: {predicted_label}")

            # Reset timer
            start_time = time.time()

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ISL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
