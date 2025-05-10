import cv2
import mediapipe as mp
import tensorflow as tf
import pickle
import numpy as np

# Load the pre-trained model and label encoder
model = tf.keras.models.load_model('word_model.keras')

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Set up MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to preprocess hand landmarks into an image format for model input
def preprocess_landmarks(landmarks, image_size=(64, 64)):
    # Create a blank image (black image)
    image = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

    # Scale the landmarks to fit within the image size
    h, w = image_size
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        # Draw a small circle at each landmark position
        cv2.circle(image, (x, y), 3, 255, -1)

    # Reshape the image to (64, 64, 1) and normalize to the range [0, 1]
    image = image.reshape((image_size[0], image_size[1], 1)).astype(np.float32) / 255.0
    return image

# Start processing video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the hand landmarks
    result = hands.process(rgb_frame)

    # Check if any hands are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Preprocess the landmarks into an image
            input_data = preprocess_landmarks(hand_landmarks.landmark)  # Process the landmarks into image format

            # Make predictions using the model
            prediction = model.predict(np.expand_dims(input_data, axis=0))  # Add batch dimension
            predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
            print(predicted_label)

            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Gesture Recognition", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()