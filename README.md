# alreadyEatenHalf
OpenSourceSW Term-Project   
Team 43: 반은 먹었조

## Members
- 202035509 Kim Yeeun
- 201934222 Kim Taehyun
- 202135751 Kim Jinha
- 202334481 Seo Beomchang

## Our Project
Detect Number of People in (Realtime) Videos using HOG Descriptor, YOLOv3 and Haarcascade Classifier

To use our code, please download pretrained weight of YOLOv3 model first. link [pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)


Then, please run only ```main.py```.


### directory

```
codes : functions and classes to use our 'main' code
ㄴ class_person_detect.py
ㄴ main.py
ㄴ people_detect_video.py
ㄴ person_detection.py
ㄴ test_tools.ipynb : show result of person_detection's functions

data : test data (image and video)
ㄴ Lenna.png
ㄴ sample.mp4
ㄴ sample_video1.mp4
ㄴ sample_video2.mp4
ㄴ test_image01.jpg
ㄴ test_image02.jpeg

tool : haarcascades xmls, configuration of YOLOv3
ㄴ coco.names
ㄴ haarcascade_frontalface_default.xml
ㄴ haarcascade_fullbody.xml
ㄴ yolov3.cfg
ㄴ yolov3.weights
```

---
### used source
- opencv haarcascades xml [opencv github](https://github.com/opencv/opencv/tree/master/data/haarcascades)
- COCO names and configuration file of YOLOv3 : [Darknet - YOLOv3](https://github.com/pjreddie/darknet/tree/master/data)


- test-Image01.jpg : [가천대학교 뉴스](https://www.gachon.ac.kr/pr/1443/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGcHIlMkY0NjQlMkY5MDMzOSUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2c3JjaENvbHVtbiUzRCUyNnNyY2hXcmQlM0QlMjZiYnNDbFNlcSUzRCUyNmJic09wZW5XcmRTZXElM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2aXNWaWV3TWluZSUzRGZhbHNlJTI2cGFzc3dvcmQlM0QlMjY%3D)
- test-Image02.jpg : [가천대학교 뉴스](https://www.gachon.ac.kr/pr/1443/subview.do?enc=Zm5jdDF8QEB8JTJGYmJzJTJGcHIlMkY0NjQlMkY5MTE3MyUyRmFydGNsVmlldy5kbyUzRnBhZ2UlM0QxJTI2c3JjaENvbHVtbiUzRCUyNnNyY2hXcmQlM0QlMjZiYnNDbFNlcSUzRCUyNmJic09wZW5XcmRTZXElM0QlMjZyZ3NCZ25kZVN0ciUzRCUyNnJnc0VuZGRlU3RyJTNEJTI2aXNWaWV3TWluZSUzRGZhbHNlJTI2cGFzc3dvcmQlM0QlMjY%3D)

- sample_video1.mp4 : [getty images](https://www.gettyimages.com/detail/video/meet-at-the-bus-stop-stock-footage/1409936094?adppopup=true)
- sample_video2.mp4 : [getty images](https://www.gettyimages.com/detail/video/waiting-at-the-bus-stop-stock-footage/1409935022)
- sample.mp4 : [Tourist Crossing The Street](https://www.pexels.com/video/tourist-crossing-the-street-855565/)