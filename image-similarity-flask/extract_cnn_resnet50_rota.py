import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from numpy import linalg as LA
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from utils import rotate_img1

class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_resnet = ResNet50(weights=self.weight, input_shape=(
            self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model_resnet.predict(np.zeros((1, 224, 224, 3)))

    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(
            self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet(img)
        feat = self.model_resnet.predict(img)
        # print(feat.shape)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
    
    def extract_feat_test(self, img_path):
        norm_feats = []
        angle = 0
        while angle < 360:
            if angle == 0:
                norm_feat = self.extract_feat(img_path)
                norm_feats.append(norm_feat)
            else:
                savePath = rotate_img1(img_path, angle)
                norm_feat = self.extract_feat(savePath)
                norm_feats.append(norm_feat)
            angle += 10
        return norm_feats