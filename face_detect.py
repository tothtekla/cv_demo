import cv2
from fer import FER
import random

# Initialize
cap = cv2.VideoCapture(0)
detector = FER(mtcnn=True)

# Predefined colors (BGR)
colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.detect_emotions(frame)

    for idx, r in enumerate(results):
        (x, y, w, h) = r["box"]

        # Assign color based on face index (cycles if more faces than colors)
        color = colors[idx % len(colors)]

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Top 2 emotions
        sorted_emotions = sorted(r["emotions"].items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_score = sorted_emotions[0]
        second_emotion, second_score = sorted_emotions[1]

        # Display emotions
        cv2.putText(frame, f"{top_emotion} ({top_score:.2f})", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        #cv2.putText(frame, f"{second_emotion} ({second_score:.2f})", (x, y - 45),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()