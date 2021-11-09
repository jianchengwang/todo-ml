import numpy as np
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

from keras.preprocessing import image
from numpy import linalg as LA

from bcnn_utils import build_model

class BCNNNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.model = build_model([224, 224, 3])
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(
            self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_vgg(img)
        feat = self.model.predict(img)
        # print(feat.shape)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat