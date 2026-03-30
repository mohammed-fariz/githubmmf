git import cv2
from ultralytics import YOLO

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = YOLO("models/truckbedyolomodel.pt")

# ==============================
# START CAMERA
# ==============================
cap = cv2.VideoCapture(0)   # 0 = webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==============================
    # RUN INFERENCE
    # ==============================
    results = model(frame, conf=0.5)

    # ==============================
    # DRAW BOUNDING BOX
    # ==============================
    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Label
            cv2.putText(
                frame,
                f"truck_bed {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    # ==============================
    # SHOW OUTPUT
    # ==============================
    cv2.imshow("YOLO Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()