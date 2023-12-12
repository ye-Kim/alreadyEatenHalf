import cv2
import numpy as np
import matplotlib.pyplot as plt 

class PersonDetection:
    def __init__(self):
        self.fullbody_xml = '../tool/haarcascade_fullbody.xml'
        self.face_xml = '../tool/haarcascade_frontalface_default.xml'
        self.body_cascade = cv2.CascadeClassifier(self.fullbody_xml)
        self.face_cascade = cv2.CascadeClassifier(self.face_xml)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def person_detection_body(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        body = self.body_cascade.detectMultiScale(gray, 1.01, 6, minSize=(30, 30))

        num_people = len(body)

        for (bx, by, bw, bh) in body:
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

        if img_show: # plot img with rectangle
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Result of Detection")
            plt.xticks([]), plt.yticks([])
            plt.show()

        return num_people, img

    def person_detection_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = self.face_cascade.detectMultiScale(gray, 1.01, 6, minSize=(30, ~30))

        num_people = len(face)

        for (fx, fy, fw, fh) in face:
            cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        if img_show: # plot img with rectangle
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Result of Detection")
            plt.xticks([]), plt.yticks([])
            plt.show()

        return num_people, img

    def person_detection_hog(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected, _ = self.hog.detectMultiScale(gray)

        for (x, y, w, h) in detected:
            cv2.rectangle(img, (x, y, w, h), (0, 0, 255), 3)

        num_people = len(detected)


        if img_show: # plot img with rectangle
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title("Result of Detection")
            plt.xticks([]), plt.yticks([])
            plt.show()

        return num_people, img

def people_detect_video(video_path, detect_type):

    cap = cv2.VideoCapture(video_path)

    # to set result video
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./data/output.avi', fourcc, fps, (w, h))
 

    person_detector = PersonDetection()

    while cap.isOpened():
        ret,frame = cap.read()
        new_frame= None
        num_people= 0
        if ret:
            if detect_type == 'face':
                (num_people,new_frame) = person_detection.person_detection_face(frame,False)
            elif detect_type == 'body':
                (num_people,new_frame) = person_detection.person_detection_body(frame,False)
            elif detect_type == 'YOLO':
                (num_people,new_frame) = person_detection.person_detection_YOLO(frame,False)
            text= "detected count : %d" %num_people
            cv2.putText(new_frame, text, (30, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0),5)
            out.write(new_frame)
    cap.release()
    out.release()
    return './data/output.avi'

def main(video_path):
    face_result = people_detect_video(video_path, detect_type='face')
    body_result = people_detect_video(video_path, detect_type='body')
    hog_result = people_detect_video(video_path, detect_type='HOG')

    return face_result, body_result, hog_result

if __name__ == "__main__":
    video_path = './data/sample.mp4'
    face_result, body_result, hog_result = main(video_path)
    print("Face Detection Result Video:", face_result)
    print("Body Detection Result Video:", body_result)
    print("HOG Detection Result Video:", hog_result)
