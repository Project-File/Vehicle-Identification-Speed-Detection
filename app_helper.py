import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import imageio

def get_image(image_path , img_name):
    classes_path = './data/labels/coco.names'
    weights_path = './weights/yolov3.tf'
    tiny = False                    
    size = 416                     
    output_path = './static/detections/'
    num_classes = 80

    # load in weights and classes
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights_path).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(classes_path).readlines()]
    print('classes loaded')
    #reading the images & apply detection with loaded weight file
    image = imageio.imread(image_path)
    filename = img_name
    img_raw = tf.image.decode_image(
        open(image_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    print('detections:')
    for i in range(nums[0]):
        print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + '{}' .format(filename), img)
    print('output saved to: {}'.format(output_path +  '{}'.format(filename)))