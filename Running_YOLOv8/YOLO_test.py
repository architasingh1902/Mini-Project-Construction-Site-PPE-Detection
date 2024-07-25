from ultralytics import YOLO
import cv2 #for showing image like resize etc

model=YOLO('C:/Users/ARCHITA SINGH/Desktop/Mini Project Vaibhav/pythonProject/YOLO-Weights/best.pt')
results=model("C:/Users/ARCHITA SINGH/Desktop/Mini Project Vaibhav/pythonProject/Images/1.png", show=True)

cv2.waitKey(0) #for delay(wait key)