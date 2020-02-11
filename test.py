'''This code is for Inference
 author : Hyunah Oh
 data : 2020.02.11
'''

import numpy as np
import cv2
import os

from keras.models import load_model
from model import create_model
from align import AlignDlib

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

#### Detection & Alignment & Normalization ####
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img), 
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

#### Embedding ####
def embedding_img(img):
    img = (img / 255.).astype(np.float32)
    return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

model = load_model('models/best1.h5')

#### Detection ####
def detect_img():
    person=['hyunjin', 'irene', 'seulgi', 'taeyun']
    img = input('Input image filename:')
    try:
        image = load_image(img)
    except:
        print('Open Error!')
    else:
        a_image = align_image(image)
        a_image = (a_image / 255.).astype(np.float32)
        embedded = nn4_small2_pretrained.predict(np.expand_dims(a_image, axis=0))
        predict = model.predict_classes(embedded)[0]
        print('Output:', person[predict])

if __name__ == '__main__':
    detect_img()