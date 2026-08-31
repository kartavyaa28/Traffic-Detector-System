import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture("traffic.mp4") 


fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


line_y = 450
offset = 6  
vehicle_count = 0

detections = []


def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    roi = frame[200:600, 100:800]

    mask = fgbg.apply(roi)
    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
 
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(roi, (0, line_y - 200), (800, line_y - 200), (0, 0, 255), 2)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        if w < 40 or h < 40:
            continue

        center = get_center(x, y, w, h)
        detections.append(center)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in detections:
        if (line_y - offset) < (y + 200) < (line_y + offset):
            vehicle_count += 1
            detections.remove((x, y))

        cv2.putText(roi, f"Vehicles: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  
    if vehicle_count < 10:
        level = "LOW"
        color = (0, 255, 0)
    elif vehicle_count < 25:
        level = "MEDIUM"
        color = (0, 255, 255)
    else:
        level = "HIGH"
        color = (0, 0, 255)

    cv2.putText(roi, f"Congestion: {level}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Traffic Detector", roi)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
