import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

fullbody_xml = '../tool/haarcascade_fullbody.xml'
face_xml = '../tool/haarcascade_frontalface_default.xml'

def person_detection_body(img, img_show=False):
    body_cascade = cv2.CascadeClassifier(fullbody_xml)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    body = body_cascade.detectMultiScale(gray, 1.01, 6, minSize=(30, 30))
    
    num_people = len(body)

    for (bx, by, bw, bh) in body:
        cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)

    if img_show: # plot img with rectangle
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Result of Detection")
        plt.xticks([]), plt.yticks([])
        plt.show()

    return num_people, img

def person_detection_face(img, img_show=False):
    face_cascade = cv2.CascadeClassifier(face_xml)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray, 1.01, 6, minSize=(30, 30))
    
    num_people = len(face)

    for (fx, fy, fw, fh) in face:
        cv2.rectangle(img, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

    if img_show: # plot img with rectangle
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Result of Detection")
        plt.xticks([]), plt.yticks([])
        plt.show()

    return num_people, img

def person_detection_hog(img, img_show=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    detected, _ = hog.detectMultiScale(gray)

    for (x, y, w, h) in detected:
        cv2.rectangle(img, (x, y, w, h), (0, 0, 255), 3)

    num_people = len(detected)
    
    if img_show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("HOG image")
        plt.show()

    return num_people, img

def person_detection_YOLO(img, img_show=False):
    num_people = 0
    height, width, channel = img.shape

    # get blob from image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # read coco object names
    with open("../tool/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # load pre-trained yolo model from configuration and weight files
    net = cv2.dnn.readNetFromDarknet('../tool/yolov3.cfg', '../tool/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # set output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # detect objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get bounding boxes and confidence socres
    class_ids = []
    confidence_scores = []
    boxes = []

    for out in outs: # for each detected object

        for detection in out: # for each bounding box

            scores = detection[5:] # scores (confidence) for all classes
            class_id = np.argmax(scores) # class id with the maximum score (confidence)
            confidence = scores[class_id] # the maximum score

            if confidence > 0.5:
                # bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidence_scores.append(float(confidence))
                class_ids.append(class_id)

    # non maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, 0.5, 0.4)

    # draw bounding boxes with labels on image
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'person':
                num_people = num_people + 1

            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
            cv2.putText(img, label, (x, y - 10), font, 3, color, 2)


    if img_show:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("YOLO image")
        plt.show()

    return num_people, img