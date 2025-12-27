import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

time.sleep(2)
print("Webcam initialized successfully")

# Capture background
frames = []
print("Capturing background... Please move out of the frame.")
for i in range(60):  
    ret, frame = camera.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frames.append(frame.astype(np.float32))
    cv2.imshow("Background", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

background = np.mean(frames, axis=0).astype(np.uint8)
cv2.destroyWindow("Capturing Background")
print("Background captured successfully")

# Define black cloak range in HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 50])

print("Start wearing cloak. Press 'q' to exit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for black color
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask_inv = cv2.bitwise_not(mask)

    # Segment out cloak area and replace with background
    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Invisibility Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
