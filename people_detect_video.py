import person_detection
import cv2

def people_detect_video(video_path,detect_type):
    # read videos
    cap = cv2.VideoCapture(video_path)
    
    # to set result video
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('./data/output.avi', fourcc, fps, (w, h))
    delay = round(1000/fps)
    
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
            
