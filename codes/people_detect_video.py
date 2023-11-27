import cv2

video_file = '../data/sample_video1.mp4'

cap = cv2.VideoCapture(video_file)

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, 1.01, 4, minSize=(70,100))

        for(x, y, w, h) in bodies:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

        print('number of people', len(bodies))

        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
