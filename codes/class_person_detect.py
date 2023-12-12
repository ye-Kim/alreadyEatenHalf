from person_detection import person_detection_body, person_detection_face, person_detection_hog, person_detection_YOLO 
from people_detect_video import people_detect_video 
import cv2 

class PersonDetection:
    def __init__(self):
        self.fullbody_xml = '../tool/haarcascade_fullbody.xml'
        self.face_xml = '../tool/haarcascade_frontalface_default.xml'
        self.body_cascade = cv2.CascadeClassifier(self.fullbody_xml)
        self.face_cascade = cv2.CascadeClassifier(self.face_xml)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def person_detection_body(self, img):
        num_people, img = person_detection_body(img, True)
        return num_people, img

    def person_detection_face(self, img):
        num_people, img = person_detection_face(img, True)
        return num_people, img

    def person_detection_hog(self, img):
        num_people, img = person_detection_hog(img, True)
        return num_people, img

    def person_detection_YOLO(self, img):
        num_people, img = person_detection_YOLO(img, True)
        return num_people, img 

    def people_detect_video(video_path, detect_type):
        res = people_detect_video(video_path, detect_type)
        return res 
