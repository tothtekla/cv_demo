import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("data/yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Predefined colors (BGR) - same as in emotion detection
colors = [
    (0, 255, 0),    # Green
    (0, 0, 255),    # Red
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
]

# Keep track of color per class
class_colors = {}

# COCO class names
class_names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"

        # Assign color from predefined list
        if cls not in class_colors:
            class_colors[cls] = colors[len(class_colors) % len(colors)]
        color = class_colors[cls]

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 Webcam - Emotion Colors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
