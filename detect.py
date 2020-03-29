import cv2
import time
import argparse
import numpy as np
from threading import Thread

class ObjectDetection():
    
    def __init__(self, classses, net):
        self.classes = classes
        self.net = net
        self.stop = False
        self.cap = cv2.VideoCapture(args['video'])


    def read(self):
        while True:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                break

            cv2.imshow("yolov3 object detection", self.frame)
            if cv2.waitKey(1) == 27:
                self.stop=True
                break
            
        self.cap.release()
        cv2.destroyAllWindows()

    def detect(self):
        # if image is passed as argument
        if args['image']:
            image = cv2.imread(args['image'])
            image = cv2.resize(image, (640, 480))
            width = image.shape[1]
            height = image.shape[0]

            # pass image to the blob and set as input to the model
            blob = cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False)
            self.net.setInput(blob)
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            # pass image to the model
            outputs = net.forward(output_layers)

            # loop over the outputs
            for out in outputs:
                for detection in out:
                    # prob score
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # box coordinates
                        w = round(int(detection[2] * width))
                        h = round(int(detection[3] * height))
                        x = round(int(detection[0] * width) - w / 2)
                        y = round(int(detection[1] * height) - h / 2)

                        label = str(self.classes[class_id])
                        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                        cv2.putText(image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("yolov3 object detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # if video file is passed or no argument is passed 
        # by default it will start live stream of cam 0
        else:
            Thread(target=self.read, args=()).start()
            time.sleep(2.0)
            while True:
                if not self.stop:
                    frame = cv2.resize(self.frame, (640, 480))
                    width = frame.shape[1]
                    height = frame.shape[0]

                    # pass image to the blob and set as input to the model
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True, crop=False)
                    self.net.setInput(blob)
                    layer_names = net.getLayerNames()
                    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    # pass image to the model
                    outputs = net.forward(output_layers)

                    # loop over the outputs
                    for out in outputs:
                        for detection in out:
                            # prob score
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                # box coordinates
                                w = round(int(detection[2] * width))
                                h = round(int(detection[3] * height))
                                x = round(int(detection[0] * width) - w / 2)
                                y = round(int(detection[1] * height) - h / 2)

                                label = str(self.classes[class_id])
                                print(label)
                                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                                cv2.putText(self.frame, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    break

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', type=str, required=False,
                    help='path to input image')
    ap.add_argument('-v', '--video', type=str, default=0,
                    help='path to video file')
    args = vars(ap.parse_args())

    try:
        # read class file that consists of all the object names
        with open('classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # load yolo weights and config file
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        # main obj
        camObj = ObjectDetection(classes, net)
        camObj.detect()
    except Exception as e:
        print(str(e))