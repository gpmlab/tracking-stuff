from imageai.Detection import ObjectDetection
import cv2
from PIL import Image
import os
import numpy as np
import io
import time


def resize(img, x, y):
    resized = cv2.resize(img, (x, y))
    return resized


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, 'resnet50_coco_best_v2.0.1.h5'))
detector.loadModel()

path = 'IMG_0858.mov'
# path = 0
cap = cv2.VideoCapture(path)

while cap.isOpened():
    ret, frame = cap.read()
    np_image = resize(frame, 640, 360)

    start_time = time.time()
    detections = detector.detectObjectsFromImage(input_image=np_image, input_type='array')
    end_time = time.time()

    num_detections = len(detections)

    print('num detected: ' + str(num_detections))
    print('took ' + str((end_time - start_time) * 1000) + 'ms')

    if num_detections:

        for eachObject in detections:
            print(eachObject["name"], " : ", eachObject["percentage_probability"])

    cv2.imshow('test', np_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
