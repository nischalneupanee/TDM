import cv2
from ultralytics import YOLO

# Load the pretrained OpenImageV7 model
model = YOLO('yolov8x-oiv7.pt')  # Replace with your pretrained model path

# Define class numbers based on your YAML file
TIGER_CLASS = 534  # Replace with the actual tiger class number from your YAML file
HUMAN_CLASS = 264 # Replace with the actual human class number from your YAML file

# Open the live camera feed
cap = cv2.VideoCapture(0)  # Use 0 or another index for your camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model(frame)

    # Process the results
    for detection in results[0].boxes:
        class_id = int(detection.cls[0])  # Get class ID for this detection
        confidence = detection.conf[0]  # Get confidence score
        x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates

        if class_id == HUMAN_CLASS:
            # Draw bounding box for human (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Human: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        elif class_id == TIGER_CLASS:
            # Draw bounding box for tiger (Red)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Tiger: {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame with detections
    cv2.imshow('Live Camera - Tiger and Human Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
