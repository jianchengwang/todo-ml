import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications import resnet

from keras.preprocessing import image
from numpy import linalg as LA


def siamese_extract_feat(img_path):
    input_shape = (200, 200, 3)
    img = image.load_img(img_path, target_size=(
        input_shape[0], input_shape[1]))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = resnet.preprocess_input(img)
    return img


cosine_similarity = metrics.CosineSimilarity()
network = tf.keras.models.load_model("D://workspace//ml//ml-models//siamese_network.h5")
a1 = network(siamese_extract_feat('1.jpg'))
a2 = network(siamese_extract_feat('2.jpg'))
a_similarity = cosine_similarity(a1, a2)
print("a_similarity:", a_similarity.numpy())

b1 = network(siamese_extract_feat('a1.jpg'))
b2 = network(siamese_extract_feat('a3.jpg'))
c1 = network(siamese_extract_feat('b1.jpg'))
b_similarity = cosine_similarity(b1, b2)
print("b_similarity:", b_similarity.numpy())
bc_similarity = cosine_similarity(b1, c1)
print("bc_similarity:", bc_similarity.numpy())
