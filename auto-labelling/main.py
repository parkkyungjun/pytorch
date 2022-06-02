import cv2
import time
import numpy as np
from tracker import *

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("test_Trim.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

delay_time = 3
prev = 0
ret = True
img_cnt = 0
while ret:
    time_elapsed = time.time() - prev
    ret, frame = cap.read()
    if time_elapsed > delay_time:
        prev = time.time()
        height, width, _ = frame.shape

        # Extract Region of interest
        # roi = frame[340: 720,500: 800]

        # 1. Object Detection
        # mask = object_detector.apply(roi)
        mask = object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            # Calculate area and remove small elements
            area = cv2.contourArea(cnt)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        if detections:
            cv2.imwrite(f'../coco/images/{img_cnt}.png', frame)

        # 2. Object Tracking
        # boxes_ids = tracker.update(detections)
        box_data = []
        for box_id in detections:
            x, y, w, h = box_id
            cv2.putText(frame, str(round((x+ w/2)/width, 6)), (x, y+100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # print('rgb', np.mean(frame[:, :, 0][y:y+h, x:x+w]), np.mean(frame[:, :, 1][y:y+h, x:x+w]) , np.mean(frame[:, :, 2][y:y+h, x:x+w]))
            # print('rgb', np.mean(frame[:, :, 1][y:y+h, x:x+w]), round(x/w, 6))

            if np.mean(frame[:, :, 1][y:y+h, x:x+w]) > 145:
                cls = 5
            else:
                cls = 8
            box_data.append([x, y, w ,h , cls])

        # cv2.imshow("roi", roi)
        if detections:
            tracker.save_txt(box_data, width, height, img_cnt)
            cv2.imwrite(f'../coco/labels/{img_cnt}_check.png', frame)
            img_cnt += 1
        cv2.imshow("Frame", frame)
        # cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()