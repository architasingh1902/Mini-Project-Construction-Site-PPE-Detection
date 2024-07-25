from ultralytics import YOLO
import cv2  # For capturing and displaying video
import math

# Initialize video capture from the default webcam
cap = cv2.VideoCapture(0)

# Get the width and height of the video frames
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Create a VideoWriter object to save the output video
# 'output.avi' is the name of the output file
# cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') specifies the codec
# 10 is the frame rate
# (frame_width, frame_height) specifies the frame size
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

# Load the YOLOv8 model from the specified path
model = YOLO("../YOLO-Weights/best.pt")

# List of class names that the YOLO model can detect
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Start a loop to read frames from the webcam
while True:
    success, img = cap.read()
    if not success:
        break

    # Perform object detection using YOLOv8 on the current frame
    results = model(img, stream=True)

    # Loop through the detection results
    for r in results:
        boxes = r.boxes  # Get the bounding boxes for detected objects
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get the coordinates for each box
            # Convert coordinates from tensor to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)  # Print coordinates to the console
            # Draw a rectangle around the detected object
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Get the confidence score of the detection
            conf = math.ceil((box.conf[0] * 100)) / 100  # Convert to integer with 2 decimal precision
            cls = int(box.cls[0])  # Get the class ID
            class_name = classNames[cls]  # Get the class name from class ID
            label = f'{class_name} {conf}'  # Create a label with the class name and confidence score

            # Get the size of the text box
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            # Calculate the coordinates for the text box
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            # Draw a filled rectangle for the text background
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            # Put the text label on the rectangle
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    #Write the frame with detections to the output video
    out.write(img)
    # Display the frame with detections
    cv2.imshow("Image", img)
    # Break the loop if '1' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

# Release the video capture and writer objects
out.release()
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
