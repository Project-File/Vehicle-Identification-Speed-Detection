from __future__ import division
import cv2
import os
import numpy as np
from PIL import Image

class Channel_value:
    val = -1.0
    intensity = -1.0

def atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = dark_channel_find(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break
    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t
    return max_channel.intensity

#Finding the dark channel i.e. the pixel with the lowest R/G/B value
def dark_channel_find(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]

#Finding a coarse image which gives us a transmission map
def coarse(minimum, x, maximum):
    return max(minimum, min(x, maximum))

#Uses values from other functions to aggregate and give us a clear image
def dehaze(img, light_intensity, windowSize, t0, w):
    size = (img.shape[0], img.shape[1])

    outimg = np.zeros(img.shape, img.dtype)

    for y in range(size[0]):
        for x in range(size[1]):
            x_low = max(x-(windowSize//2), 0)
            y_low = max(y-(windowSize//2), 0)
            x_high = min(x+(windowSize//2), size[1])
            y_high = min(y+(windowSize//2), size[0])

            sliceimg = img[y_low:y_high, x_low:x_high]

            dark_channel = dark_channel_find(sliceimg)
            t = 1.0 - (w * img.item(y, x, dark_channel) / light_intensity)

            outimg.itemset((y,x,0), coarse(0, ((img.item(y,x,0) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,1), coarse(0, ((img.item(y,x,1) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,2), coarse(0, ((img.item(y,x,2) - light_intensity) / max(t, t0) + light_intensity), 255))
    return outimg

# Commented out IPython magic to ensure Python compatibility.
# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    d="./color_short"
    cd="./color_short_output"
    folder=os.listdir(d) 
    for fold in folder:
        # print(fold)
        files=os.listdir(os.path.join(d,fold))
        for f in files:
            # print(f)
            img = cv2.imread(os.path.join(d,fold,f))
            imgplot = plt.imshow(img)
            plt.show()
            
            img = np.array(img, dtype=np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            light_intensity = atmospheric_light(img, gray)
            w = 0.95
            t0 = 0.55
            outimg = dehaze(img, light_intensity, 20, t0, w)
            name = os.path.join(cd,fold,f)
            imgplot = plt.imshow(outimg)
            plt.show()
            # print(name)
            cv2.imwrite(name, outimg)

# main()

import matplotlib.pyplot as plt
from PIL import Image

class Colors(object):
    class Color(object):
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return "%s : %s" % (self.__class__.__name__, self.value)

    class Red(Color): pass
    class Blue(Color): pass
    class Green(Color): pass
    class Yellow(Color): pass
    class White(Color): pass
    class Gray(Color): pass
    class Black(Color): pass
    class Pink(Color): pass
    class Teal(Color): pass

class ColorWheel(object):
    def __init__(self, rgb):
        r, g, b = rgb

        self.rgb = (Colors.Red(r), Colors.Green(g), Colors.Blue(b), )
        

    def estimate_color(self):
        dominant_colors = self.get_dominant_colors()

        total_colors = len(dominant_colors)

        if total_colors == 1:
            return dominant_colors[0]
        elif total_colors == 2:
            color_classes = [x.__class__ for x in dominant_colors]

            if Colors.Red in color_classes and Colors.Green in color_classes:
                return Colors.Yellow(dominant_colors[0].value)
            elif Colors.Red in color_classes and Colors.Blue in color_classes:
                return Colors.Pink(dominant_colors[0].value)
            elif Colors.Blue in color_classes and Colors.Green in color_classes:
                return Colors.Teal(dominant_colors[0].value)
        elif total_colors == 3:
            if dominant_colors[0].value > 200:
                return Colors.White(dominant_colors[0].value)
            elif dominant_colors[0].value > 100:
                return Colors.Gray(dominant_colors[0].value)
            else:
                return Colors.Black(dominant_colors[0].value)
        else:
            print("Dominant Colors : %s" % dominant_colors)

    def get_dominant_colors(self):
        max_color = max([x.value for x in self.rgb])

        return [x for x in self.rgb if x.value >= max_color * .9]

def process_image(image):
    d={}
    image_color_quantities = {}
    width, height = image.size
    # print(image.size)


    width_margin = int(width - (width * .65))
    height_margin = int(height - (height * .65))
    # print(width_margin,height_margin)
    for x in range(width_margin, width - width_margin):
        for y in range(height_margin, height - height_margin):
            r, g, b = image.getpixel((x, y))

            key = "%s:%s:%s" % (r, g, b, )

            key = (r, g, b, )

            image_color_quantities[key] = image_color_quantities.get(key, 0) + 1

    total_assessed_pixels = sum([v for k, v in image_color_quantities.items() if v > 10])

    strongest_color_wheels = [(ColorWheel(k), v / float(total_assessed_pixels) * 100, ) for k, v in image_color_quantities.items() if v > 10]
    final_colors = {}

    for color_wheel, strength in strongest_color_wheels:
        color = color_wheel.estimate_color()

        final_colors[color.__class__] = final_colors.get(color.__class__, 0) + strength

    for color, strength in final_colors.items():
        d[color.__name__]=strength
    return d

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_car_colour(path):
    # image = cv2.imread(path)
    # img = np.array(image, dtype=np.uint8)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # light_intensity = atmospheric_light(img, gray)
    # w = 0.95
    # t0 = 0.55
    # outimg = dehaze(img, light_intensity, 20, t0, w)
    # op_folder = "./color_short_output/green"
    # cv2.imwrite(os.path.join(op_folder, 'op_img.jpg'), outimg)
    # outimg = Image.open(os.path.join(op_folder, 'op_img.jpg'))
    # d = process_image(outimg)

    image = Image.open(path)
    d = process_image(image)

    # print(d)

    colour = max(d, key=d.get)
    return(colour)
