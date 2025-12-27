import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

time.sleep(2)
print("Webcam initialized successfully")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("Invisibility Cloak", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
