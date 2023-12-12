from codes.class_person_detect import PersonDetection 

def main(video_path):
    face_result = PersonDetection.people_detect_video(video_path, detect_type='face')
    body_result = PersonDetection.people_detect_video(video_path, detect_type='body')
    hog_result = PersonDetection.people_detect_video(video_path, detect_type='hog')
    yolo_result = PersonDetection.people_detect_video(video_path, detect_type='YOLO')

    return face_result, body_result, hog_result, yolo_result

if __name__ == "__main__":
    video_path = './data/sample.mp4'

    face_result, body_result, hog_result, yolo_result = main(video_path)
    print("Face Detection Result Video:", face_result)
    print("Body Detection Result Video:", body_result)
    print("HOG Detection Result Video:", hog_result)
    print("YOLO Detection Result Video:", yolo_result)
