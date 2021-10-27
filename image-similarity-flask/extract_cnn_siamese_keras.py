import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications import resnet

from keras.preprocessing import image
from numpy import linalg as LA

import utils

class SIAMESENet:
    def __init__(self):
        self.input_shape = (200, 200, 3)
        self.siamese_network = tf.keras.models.load_model(os.path.join(utils.MODELS, "siamese_network.h5"))

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(
            self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet(img)
        feat = self.siamese_network.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

