from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("D:\objDetection\Videos\cartraffic1.mp4")



model = YOLO('..\Yolo-weights\yolov8l.pt')

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat ", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone" "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask_path = "D:\objDetection\Car Counter\mask.jpg"
mask = cv2.imread(mask_path)
print("mask", mask)

#tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [900,700,1500,700]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    result = model(imgRegion, stream = True)
    print(result)

    detections = np.empty((0,5))
    for r in result:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,255), 3)
            w,h = x2-x1, y2-y1

            conf = math.ceil((box.conf[0]*100))/100
 
            cls = int(box.cls[0])
            print('cls', cls)
            print("name", className[cls])
            currentClass = className[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf>0.3:
                #cvzone.putTextRect(img, f'{className[cls]}',(max(0,x1),max(35,y1)), scale=0.6, thickness= 1, offset = 3)
                cvzone.cornerRect(img,(x1,y1,w,h), l =9, rt =5)
                currentArray = np.array([x1,y1,x2,y2, conf])
                detections = np.vstack((detections,currentArray))
    
    resultTracker = tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    for res in resultTracker:
        x1,y1,x2,y2,id = res
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(res)
        cvzone.cornerRect(img, (x1,y1,w,h), l =9, rt =2, colorR = (255,0,0))
        cvzone.putTextRect(img, f'{id}',(max(0,x1),max(35,y1)), scale=0.6, thickness= 1, offset = 3)

        cx,cy = x1+w//2 ,y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0]<cx< limits[2] and limits[1] - 40 <cy< limits[1] + 40:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)
    
    cvzone.putTextRect(img, f'count: {totalCount}',(50,50))


    cv2.imshow("Image",img)
    #cv2.imshow("ImgRegion", imgRegion)
    cv2.waitKey(1)