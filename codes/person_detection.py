import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

fullbody_xml = '../tool/haarcascade_fullbody.xml'
face_xml = '../tool/haarcascade_frontalcatface_extended.xml'

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