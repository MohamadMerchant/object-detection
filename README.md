## YOLO OBJECT DETECTION ON IMAGES, VIDEOS and LIVE STREAM

### Steps to run on Images, Videos or Live stream


- **`git clone https://github.com/MohamadMerchant/object-detection.git**`**

- **cd yolo-object-detection**

- 	`*pip3 install -r requirements.txt*`

- Download the pre-trained YOLO v3 weights file from this [link](https://pjreddie.com/media/files/yolov3.weights) 
  or you can directly download to the current directory in terminal using
 
 	`wget https://pjreddie.com/media/files/yolov3.weights`


**To detect objects on image**

`*python3 detect.py --image IMAGE_PATH*`

**To detect objects on video file**

`*python3 detect.py --video VIDEO_PATH*`

**To detect objects on a live stream**

`*python3 detect.py*`
