import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Canvas for drawing
canvas = None

# Open webcam
cap = cv2.VideoCapture(0)

# Previous coordinates of index finger
prev_x, prev_y = 0, 0

# Brush thickness
brush_thickness = 5

# Drawing state
drawing = False

# Time reference for rainbow
start_time = time.time()

print("Press 's' to start drawing, 'd' to finish the current line, 'c' to clear canvas, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror image

    if canvas is None:
        canvas = frame.copy() * 0  # black canvas

    # Convert BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(frame_rgb)

    # Check if index finger tip is detected
    finger_detected = False
    x, y = 0, 0
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)
            finger_detected = True

    if drawing and finger_detected:
        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y

        # Compute rainbow color based on time
        t = time.time() - start_time
        hue = int((t * 100) % 180)
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

        # Draw line on canvas
        cv2.line(canvas, (prev_x, prev_y), (x, y), color_bgr, brush_thickness)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0  # reset when not drawing or finger lost

    # Merge canvas with frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("Rainbow Air Drawing", frame)

    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = frame.copy() * 0
        print("Canvas cleared!")
    elif key == ord('s'):
        if finger_detected:
            drawing = True
            print("Started drawing...")
        else:
            print("Finger not detected. Cannot start drawing.")
    elif key == ord('f'):
        drawing = False
        prev_x, prev_y = 0, 0
        print("Line finished.")
