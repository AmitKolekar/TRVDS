import cv2
import numpy as np
import os
import imutils
from tensorflow.keras.models import load_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

net = cv2.dnn.readNet("utils/yolov3-custom_7000.weights", "utils/yolov3-custom.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
file = "TrafficRecord/bike/no-helmet/"


model = load_model('utils/helmet-nonhelmet_cnn.h5')
print('model loaded!!!')

COLORS = [(0,255,0),(0,0,255)]


def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi,dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi/255.0
        print("model prediction",model.predict(helmet_roi)[0][0])
        return int(model.predict(helmet_roi)[0][0])
    except:
            pass

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def extract_helmet(imgfile,n):

    img = cv2.imread(imgfile)
    img = imutils.resize(img,height=500)
    
    height, width = img.shape[0],img.shape[1]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []
    classIds = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                classIds.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            color = [int(c) for c in COLORS[classIds[i]]]
            helmet_roi = img[max(0,y):max(0,y)+max(0,h)//4,max(0,x):max(0,x)+max(0,w)]
            x_h = x-60
            y_h = y-350
            w_h = w+100
            h_h = h+100
            # h_r = img[max(0,(y-330)):max(0,(y-330 + h+100)) , max(0,(x-80)):max(0,(x-80 + w+130))]
            h_r = img[y_h:y_h+h_h , x_h:x_h +w_h]
            c = helmet_or_nohelmet(h_r)
            if c is None:
                c=0
            print(['helmet','no-helmet'][c],n)
            if c==1:
                cv2.imwrite(file+n+".jpg", img)               


    cv2.imshow("Image", img)

    if cv2.waitKey(0)==113:
        exit()
    cv2.destroyAllWindows()


arr = os.listdir("TrafficRecord/bike")

for i in arr:
    print(i)
    extract_helmet("TrafficRecord/bike/{}".format(i),i)

