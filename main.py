from os.path import join
import sys
import tempfile
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from yolov4.tf import YOLOv4
import cv2
from car_colour_new import get_car_colour
from speed_check_new import speed_detection
# import yolov4.common.media
import numpy as np
import pytesseract
import os
import random
import colorsys
import re
import cv2
import csv

UPLOAD_FOLDER = './static/uploads'
output_path = './static/detections/'
sec = 0
frameRate = 0.1
count = 0
csvfile = None
obj = None
frameTimeStamp = []
stateLPInitial = set(["AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL","MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS","TR","UP","UK","WB","AN","CH","DN","DD","DL","JK","LD","PY"])

_HSV = [(1.0 * x / 256, 1.0, 1.0) for x in range(256)]
_COLORS = list(map(lambda x: colorsys.hsv_to_rgb(*x), _HSV))
_COLORS = list(
    map(
        lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        _COLORS,
    )
)
BBOX_COLORS = []
_OFFSET = [0, 8, 4, 12, 2, 6, 10, 14, 1, 3, 5, 7, 9, 11, 13, 15]
for i in range(256):
    BBOX_COLORS.append(_COLORS[(i * 16) % 256 + _OFFSET[(i * 16) // 256]])


def draw_bboxes1(image: np.ndarray, bboxes: np.ndarray, classes: dict):
    """
    @parma image:  Dim(height, width, channel)
    @param bboxes: (candidates, 4) or (candidates, 5)
            [[center_x, center_y, w, h, class_id], ...]
            [[center_x, center_y, w, h, class_id, propability], ...]
    @param classes: {0: 'person', 1: 'bicycle', 2: 'car', ...}

    @return drawn_image

    Usage:
        image = media.draw_bboxes(image, bboxes, classes)
    """
    image = np.copy(image)
    height, width, _ = image.shape

    # Set propability
    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    # Convert ratio to length
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height

    # Draw bboxes
    for bbox in bboxes:
        c_x = int(bbox[0])
        c_y = int(bbox[1])
        half_w = int(bbox[2] / 2)
        half_h = int(bbox[3] / 2)
        top_left = (c_x - half_w, c_y - half_h)
        bottom_right = (c_x + half_w, c_y + half_h)
        class_id = int(bbox[4])
        bbox_color = BBOX_COLORS[class_id]
        font_size = 0.4
        font_thickness = 1

        # Draw box
        cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

        # Draw text box
        bbox_text = "{}: {:.1%}".format(classes[class_id], bbox[5])
        t_size = cv2.getTextSize(bbox_text, 0, font_size, font_thickness)[0]
        cv2.rectangle(
            image,
            top_left,
            (top_left[0] + t_size[0], top_left[1] - t_size[1] - 3),
            bbox_color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            bbox_text,
            (top_left[0], top_left[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255 - bbox_color[0], 255 - bbox_color[1], 255 - bbox_color[2]),
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    return image


class_dict = {
    0: "non-car",
    1: "non-car",
    2: "car",
    3: "non-car",
    4: "non-car",
    5: "non-car",
    6: "non-car",
    7: "non-car",
    8: "non-car",
    9: "non-car",
    10: "non-car",
    11: "non-car",
    12: "non-car",
    13: "non-car",
    14: "non-car",
    15: "non-car",
    16: "non-car",
    17: "non-car",
    18: "non-car",
    19: "non-car",
    20: "non-car",
    21: "non-car",
    22: "non-car",
    23: "non-car",
    24: "non-car",
    25: "non-car",
    26: "non-car",
    27: "non-car",
    28: "non-car",
    29: "non-car",
    30: "non-car",
    31: "non-car",
    32: "non-car",
    33: "non-car",
    34: "non-car",
    35: "non-car",
    36: "non-car",
    37: "non-car",
    38: "non-car",
    39: "non-car",
    40: "non-car",
    41: "non-car",
    42: "non-car",
    43: "non-car",
    44: "non-car",
    45: "non-car",
    46: "non-car",
    47: "non-car",
    48: "non-car",
    49: "non-car",
    50: "non-car",
    51: "non-car",
    52: "non-car",
    53: "non-car",
    54: "non-car",
    55: "non-car",
    56: "non-car",
    57: "non-car",
    58: "non-car",
    59: "non-car",
    60: "non-car",
    61: "non-car",
    62: "non-car",
    63: "non-car",
    64: "non-car",
    65: "non-car",
    66: "non-car",
    67: "non-car",
    68: "non-car",
    69: "non-car",
    70: "non-car",
    71: "non-car",
    72: "non-car",
    73: "non-car",
    74: "non-car",
    75: "non-car",
    76: "non-car",
    77: "non-car",
    78: "non-car",
    79: "non-car"
}

class_dict_lp = {
    0: "license_plate"
}

class_dict_logo = {
    0: "Suzuki",
    1: 'Hyundai',
    2: "Toyota",
    3: "Mahindra"
}

###########################Loading Weights and Model############################
yolo = YOLOv4()
yolo.classes = "./data/classes/coco.names"
yolo.make_model()
yolo.load_weights("./data/yolov4.weights", weights_type="yolo")
yolo_lp = YOLOv4()
yolo_lp.classes = "./data/classes/license.names"
yolo_lp.make_model()
yolo_lp.load_weights("./data/yolov4-license.weights", weights_type="yolo")
# yolo_lp.inference("./car_video.mp4", is_image=False,score_threshold=0.8)

yolo_logo = YOLOv4()
yolo_logo.classes = "./data/classes/logo.names"
yolo_logo.make_model()
yolo_logo.load_weights("./data/yolov4-logo.weights", weights_type="yolo")

# yolo.save_as_tflite("yolov4-tiny-float16.tflite")

# yolo_lp.save_as_tflite("yolov4_lp-tiny-float16.tflite")

# yolo_logo.save_as_tflite("yolov4_logo-tiny-float16.tflite")

def validateLPText(text):
    if len(text)<2:
        return False
    if text[:2] not in stateLPInitial:
        return False
    if not text[-1].isnumeric:
        return False
    for letter in text:
        if not letter.isnumeric() and letter.islower():
            return False
    return True

def get_top_left_bottom_right(bbox):
    c_x = int(bbox[0])
    c_y = int(bbox[1])
    half_w = int(bbox[2] / 2)
    half_h = int(bbox[3] / 2)
    top_left = (c_x - half_w, c_y - half_h)
    bottom_right = (c_x + half_w, c_y + half_h)
    return(top_left, bottom_right)


def detect_LP(input_img,filename):
    height, width, _ = input_img.shape
    bboxes = yolo_lp.predict(input_img)
    # Dim(-1, (x, y, w, h, class_id, probability))
    if len(bboxes):
        bboxes_og = np.copy(bboxes)
        # Set propability
        if bboxes.shape[-1] == 5:
            bboxes = np.concatenate(
                [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
            )
        else:
            bboxes = np.copy(bboxes)

        # Convert ratio to length
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
        for bbox in bboxes:
            if len(bbox) and bbox[5] >= 0.9:
                try:
                    top_left, bottom_right = get_top_left_bottom_right(bbox)
                    crop_height = bottom_right[1]-top_left[1]
                    crop_width = bottom_right[0]-top_left[0]

                    crop_img = input_img[top_left[1]:top_left[1] +
                                        crop_height, top_left[0]:top_left[0]+crop_width]
                    # cv2.imwrite("{}.png".format(filename), crop_img)
                    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (0, 0), fx=5, fy=5)
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    # filename = "{}.png".format(random)
                    # cv2.imwrite(filename, gray)

                    conf = r"--oem 1 --psm 6"
                    text = pytesseract.image_to_string(
                        gray, config=conf, lang='eng')
                    # os.remove(filename)
                    return(re.sub(r'\W+', '', text), bboxes_og)
                except Exception:
                    pass


def detect_logo(input_img):
    height, width, _ = input_img.shape
    bboxes = yolo_logo.predict(input_img)
    # Dim(-1, (x, y, w, h, class_id, probability))
    if len(bboxes):
        bboxes_og = np.copy(bboxes)
        if bboxes.shape[-1] == 5:
            bboxes = np.concatenate(
                [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
            )
        else:
            bboxes = np.copy(bboxes)

        # Convert ratio to length
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
        for bbox in bboxes:
            if len(bboxes[0]) and bboxes[0][5] >= 0.9:
                top_left, bottom_right = get_top_left_bottom_right(bbox)
                return(class_dict_logo[int(bboxes[0][4])], bboxes_og)


def convert_yolo_to_original_and_crop(bboxes, image, filename):
    height, width, _ = image.shape
    global csvfile,obj
    # Set propability
    if bboxes.shape[-1] == 5:
        bboxes = np.concatenate(
            [bboxes, np.full((*bboxes.shape[:-1], 1), 2.0)], axis=-1
        )
    else:
        bboxes = np.copy(bboxes)

    # Convert ratio to length
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * width
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * height
    #Dim(-1, (x, y, w, h, class_id, probability))
    # print(bboxes)
    # Draw bboxes
    for bbox in bboxes:  # iterating through bboxes of first detection
        if bbox[4] == 2 and bbox[5] >= 0.9:
            try:
                top_left, bottom_right = get_top_left_bottom_right(bbox)
                crop_height = bottom_right[1]-top_left[1]
                crop_width = bottom_right[0]-top_left[0]

                car_img = image[top_left[1]:top_left[1] +
                                crop_height, top_left[0]:top_left[0]+crop_width]
                lp_text, lp_bboxes = detect_LP(car_img,filename)
                if not validateLPText(lp_text):
                    raise Exception("")
                try:
                    temp_file = "{}.png".format(random.randint(7999, 9999))
                    cv2.imwrite("temp/"+temp_file, car_img)
                    colour = get_car_colour("temp/"+temp_file)
                    os.remove("temp/"+temp_file)
                except Exception:
                    colour = "-"
                logo, logo_bboxes = detect_logo(car_img)
                car_img = draw_bboxes1(car_img, logo_bboxes, class_dict_logo)
                car_img = draw_bboxes1(car_img, lp_bboxes, class_dict_lp)
                
                # colour=get_car_colour()
                # cv2.imshow('result',car_img)
                # cv2.waitKey()
                image[top_left[1]:top_left[1] +  crop_height, top_left[0]:top_left[0]+crop_width] = car_img
                cv2.imwrite(output_path + "data/" +'{}' .format(filename), image)
                print(filename, logo, lp_text, colour)
                frameNo = filename[5:]
                frameNo = int(frameNo[:-4])
                obj.writerow((filename, logo, lp_text, colour,frameTimeStamp[frameNo]))
                return(logo, lp_text,colour)
            except Exception:
                pass
                # cv2.imwrite(output_path +"data/"+ '{}' .format(filename), image)
                return('-', '-', '-')

def get_image_data(filepath, filename):
    global csvfile,obj
    # Dim(-1, (x, y, w, h, class_id, probability))
    framesFolder = os.path.join(UPLOAD_FOLDER, "VidToFrame")
    frameFiles = os.listdir(framesFolder)
    logo, lp_text, colour = ("-", "-", "-")
    csvfile = open(output_path+'data/output.csv', 'a', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(("Frame No.", "Manufacturer", "License Plate", "Colour","Time Stamp(s)"))
    
    for file in frameFiles:
        try:
            imgPath = os.path.join(framesFolder, file)
            input_img = cv2.imread(imgPath)
            ans = yolo.predict(input_img)
            # frame = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            logo, lp_text, colour = convert_yolo_to_original_and_crop(
                ans, input_img, file)            
            # colour = get_car_colour(imgPath)
            # print(file)
            # print(logo, lp_text, colour)
            # return(logo, lp_text, colour)
        except Exception:
            pass
            # print(Exception)
            logo, lp_text, colour = ("-", "-", "-")
            # print(logo, lp_text, colour)
    csvfile.close()
    return(logo, lp_text, colour)

vidcap=""
def getFrame(sec):
    global frameTimeStamp
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = vidcap.read()
    frameTimeStamp.append(str(sec))
    if hasFrames:
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'],"VidToFrame") +
                    "/frame%d.jpg" % count, image)
    return hasFrames

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/about")
# def about():
#     return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global sec,frameRate,count,vidcap,frameTimeStamp
    logo, lp_text, colour, filename = ("", "", "", "")
    if request.method == 'POST':
        vidToFramePath = os.path.join(UPLOAD_FOLDER,"VidToFrame")
        for filename in os.listdir(vidToFramePath):
            os.remove(os.path.join(vidToFramePath, filename))

        dataoutput = os.path.join(output_path, "data")

        for filename in os.listdir(dataoutput):
            os.remove(os.path.join(dataoutput, filename))
        
        f = request.files['file']

        filename = secure_filename(f.filename)
        print(filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)
        f.save(filepath)

        vidcap = cv2.VideoCapture(os.path.join(
            app.config['UPLOAD_FOLDER'], filename))

        success = getFrame(sec)

        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)
        vidcap.release()
        vidcap=""
        # os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        logo, lp_text, colour = get_image_data(filepath, filename)
        # print(logo, lp_text, colour)
        # logo, lp_text, colour = ("_", "_", "_") 
        return render_template("data_done.html", fname=filename, detected_logo=logo, detected_lp_text=lp_text, detected_colour=colour)

        

    return render_template("uploaded.html", fname=filename, detected_logo=logo, detected_lp_text=lp_text, detected_colour=colour)


@app.route('/speed', methods=['GET', 'POST'])
def upload_file_speed():
    filename=""
    if request.method == 'POST':
        speedPath = os.path.join(UPLOAD_FOLDER, "speed")
        for filename in os.listdir(speedPath):
            os.remove(os.path.join(speedPath, filename))
            # print(filename)
        speedoutput = os.path.join(output_path, "speed")
        for filename in os.listdir(speedoutput):
            os.remove(os.path.join(speedoutput, filename))
            # print(filename)
        f = request.files['file']

        filename = secure_filename(f.filename)
        print(filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'],"speed", filename)
        print(filepath)
        f.save(filepath)
        speed_detection(filepath)
        # logo, lp_text, colour = get_image_data(filepath, filename)
        

        # vidcap = cv2.VideoCapture(os.path.join(
        #     app.config['UPLOAD_FOLDER'], filename))

        # success = getFrame(sec)

        # while success:
        #     count = count + 1
        #     sec = sec + frameRate
        #     sec = round(sec, 2)
        #     success = getFrame(sec)
    return render_template("speed.html", fname=filename)


if __name__ == '__main__':
    app.run(port=7000, debug=True)
