import cv2
from ultralytics import YOLO

# Load the YOLOv8n model with your custom-trained 'tiger.pt'
model = YOLO('yolov8s-oiv7')  # Path to your trained model

# Open the default camera (0 for the first camera, 1 for the second, etc.)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set a confidence threshold to filter low-confidence detections
confidence_threshold = 0.5  # Adjust this value as needed

# Start reading from the camera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from the camera.")
        break
    
    # Perform YOLOv8 model inference on the current frame
    results = model.predict(source=frame, show=False)

    # Iterate through the detection results
    for detection in results[0].boxes:
        score = float(detection.conf.cpu().numpy().item())  # Confidence score
        
        # Process only if the confidence is higher than the threshold
        if score > confidence_threshold:
            box = detection.xyxy[0].cpu().numpy().astype(int)  # Bounding box coordinates
            class_id = int(detection.cls.cpu().numpy().item())  # Class ID

            # Map class ID 0 to the "Tiger" label
            if class_id == 534:
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = box

                # Draw the bounding box around the detected tiger
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box

                # Add the "Tiger" label with the confidence score on the frame
                label = f"Tiger ({score:.2f})"
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with annotations
    cv2.imshow('Live Tiger Detection', frame)

    # Press 'q' to exit the live camera feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
