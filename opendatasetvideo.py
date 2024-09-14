import cv2
from ultralytics import YOLO

# Load the pretrained OpenImageV7 model
model = YOLO('yolov8x-oiv7.pt')  # Replace with your pretrained model path

# Define class numbers based on your YAML file
TIGER_CLASS = 534 # Replace with the actual tiger class number from your YAML file
HUMAN_CLASS = 264   # Replace with the actual human class number from your YAML file

# Load the video file
video_path = './tiger.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Get the video writer initialized to save the output video
output_path = 'output_video.mp4'  # Output file path
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second of the original video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the current frame
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

    # Write the frame with detections to the output video
    out.write(frame)

    # Optionally, display the frame with detections (for visualization purposes)
    cv2.imshow('Video - Tiger and Human Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
