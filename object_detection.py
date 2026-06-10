import cv2
from ultralytics import YOLO

# Better YOLO model
model = YOLO("yolov8s.pt")

# Webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    # Detection with confidence threshold
    results = model.track(
    frame,
    persist=True,
    conf=0.5
)
    # Draw boxes + labels
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()