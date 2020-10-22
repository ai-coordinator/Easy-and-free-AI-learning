# coding:utf-8
import os
import re
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import array_to_img, img_to_array,load_img

from PIL import Image

root_dir = "./image/"
directory = os.listdir(root_dir)
categories = [f for f in directory if os.path.isdir(os.path.join(root_dir, f))]
print(categories)

num_classes = len(categories)

image_size = 224

def list_pictures(directory, ext='jpg|gif|png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def convertCIFER10Data(image):
    img = image.astype('float32')
    img /= 255
    c = np.zeros(image_size*image_size*3).reshape((1,image_size,image_size,3))
    c[0] = img
    return c

if __name__ == "__main__":

    model = load_model('./saved_models/original_trained_model.h5')
    for picture in list_pictures('./predict/'):

        image = Image.open(picture)
        image = image.resize((image_size, image_size))
        resize_frame = np.asarray(image)
        data = convertCIFER10Data(resize_frame)

        ret = model.predict(data, batch_size=1)

        print("----------------------------------------------")
        print("I think...")

        bestnum = 0.0
        bestclass = 0
        for n in range(num_classes):
            print("[{}] : {}%".format(categories[n], round(ret[0][n]*100,2)))
            if bestnum < ret[0][n]:
                bestnum = ret[0][n]
                bestclass = n

        print("probability : {}%".format( round(bestnum*100,2) ))
        print(picture,'→',"I think this is a [{}].".format(categories[bestclass]))
