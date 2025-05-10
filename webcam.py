import cv2
import time

# Open the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Set the video resolution (optional)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
out = cv2.VideoWriter('webcam_video.mp4', fourcc, 20.0, (640, 480))

# Start recording video
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame to the video file
    out.write(frame)

    # Display the webcam feed
    cv2.imshow('Webcam Feed', frame)

    # Stop after 5 seconds
    if time.time() - start_time > 5:
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Video saved as 'webcam_video.mp4'")
