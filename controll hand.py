import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load an image
image_path = "c:\\Users\\Admin\\Pictures\\ram documents\\1696125514552.jpg"  # Change to your image path
image = cv2.imread(image_path)
original_height, original_width = image.shape[:2]

# Video Capture
cap = cv2.VideoCapture(0)

scale_factor = 1  # Initial scale factor for image resizing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect for better control
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb tip (landmark 4) and index finger tip (landmark 8)
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]

            # Convert to pixel coordinates
            thumb_x, thumb_y = int(thumb.x * frame.shape[1]), int(thumb.y * frame.shape[0])
            index_x, index_y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

            # Calculate the Euclidean distance between thumb and index finger
            distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

            # Adjust scale factor based on finger distance
            scale_factor = max(0.5, min(2.0, distance / 100))  # Keeps zoom within range (0.5x to 2x)

    # Resize image based on scale factor
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Show both webcam feed and resized image
    cv2.imshow("Webcam", frame)
    cv2.imshow("Zoomed Image", resized_image)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




# #explanations
# cv2 → Used for computer vision tasks like capturing video and displaying images.
# numpy → Used for mathematical operations like calculating the Euclidean distance.
# mediapipe → A powerful library for hand tracking.



# mp.solutions.hands → Loads the hand tracking model.
# mp.solutions.drawing_utils → Used to draw hand landmarks on the video.
# Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
# min_detection_confidence → Ensures only confident detections are considered.
# min_tracking_confidence → Ensures stable tracking of hand landmarks.

# Loads an image from the specified path.
# cv2.imread(image_path) → Reads the image as a NumPy array.
# image.shape[:2] → Extracts the height and width of the image.

# Opens the default webcam (0 refers to the primary webcam).
# Used to capture real-time video feed.

# scale_factor determines how much the image should be zoomed in or out.
# Starts at 1 (original size).


# cap.isOpened() → Ensures the webcam is working.
# cap.read() → Captures each frame from the webcam.
# ret → If False, it means there is an issue (e.g., webcam disconnected).

# cv2.flip(frame, 1) → Mirrors the webcam feed for natural hand control.
# cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) → Converts frame from BGR to RGB, required by MediaPipe.

# This detects hands and landmarks in the frame.

# If hands are detected, it draws the landmarks on the frame.
# Finds the positions of:
# Thumb tip (Landmark 4)
# Index finger tip (Landmark 8)


# Mediapipe landmarks are normalized (0 to 1 range).
# Convert them to pixel values by multiplying with frame size.

# Uses Euclidean Distance Formula:
# distance
# Measures how far apart the fingers are.


# If fingers move closer, the scale factor decreases (Zoom Out).
# If fingers move apart, the scale factor increases (Zoom In).
# Keeps zoom between 0.5x (min) and 2.0x (max).


# Calculates new image size based on scale_factor.
# Uses cv2.resize() to scale the image accordingly.

# Shows the live webcam feed ("Webcam" window).
# Shows the dynamically resized image ("Zoomed Image" window).

# Listens for the 'q' key to exit the loop.


# cap.release() → Stops webcam.
# cv2.destroyAllWindows() → Closes all OpenCV windows.
