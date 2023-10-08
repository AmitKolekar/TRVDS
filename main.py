import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *


cap = cv2.VideoCapture("Videos/final_traffic.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")

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

#mask = cv2.imread("mask.png")

# Tracking
Ctracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
Btracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

entryLine = [100, 25, 1173, 25]
exitLine = [100, 300, 1173, 300]
#totalCount = []
armv = [10,14,45,54,56,62,70,91,117,133,142,143,148,152,155,
 163,167,172,179,237,253,255,257,262,268,270,272,302,306,317]
mbk_flag = []

while True:
    success, img = cap.read()
    if not success:
        break
    # imgRegion = cv2.bitwise_and(img, mask)

    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    results = model(img, stream=True)

    Bdetections = np.empty((0, 5))
    Cdetections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" and conf > 0.3:

                CcurrentArray = np.array([x1, y1, x2, y2, conf])
                Cdetections = np.vstack((Cdetections, CcurrentArray))

            if currentClass == "motorbike" and conf > 0.3:

                BcurrentArray = np.array([x1, y1, x2, y2, conf])
                Bdetections = np.vstack((Bdetections, BcurrentArray))


    resultsTracker_car = Ctracker.update(Cdetections)
    resultsTracker_bike = Btracker.update(Bdetections)

    for result in resultsTracker_car:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        s = Ctracker.getsp(int(id))
        if (Ctracker.f[int(id)] == 1):
            Ctracker.capture(img, x1, y1, h, w, s, int(id), 0, 0)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3,colorR=(0, 255, 165), offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

    for result in resultsTracker_bike:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        s = Btracker.getsp(int(id))
        if (Btracker.f[int(id)] == 1):
            Btracker.capture(img, x1, y1, h, w, s, int(id), 80, 1)

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3,colorR=(0, 255, 165), offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)


    cv2.line(img, (entryLine[0], entryLine[1]), (entryLine[2], entryLine[3]), (0, 255, 0), 3) #LINE to denote entry 
    cv2.line(img, (exitLine[0], exitLine[1]), (exitLine[2], exitLine[3]), (255, 0, 0), 3) #LINE to denote exit 

    cv2.imshow("Image", img)
    cv2.waitKey(1)
