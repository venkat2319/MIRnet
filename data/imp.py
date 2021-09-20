import scipy
import cv2
import matplotlib
import h5py
import keras

from imageai.Detection import ObjectDetection
import os
path=os.getcwd()

obj_detector= ObjectDetection()
obj_detector.setModelTypeAsRetinaNet()
obj_detector.setModelPath(os.path.join(path,"/content/MIRNet/resnet50_coco_best_v2.0.1.h5"))
#obj_detector.loadModel()
detections=obj_detector.detectObjectsFromdata(input_images=os.path.join(path,"/content/MIRNet/464.png"),output_image_path=os.path.join(path,"result.jpeg"))
