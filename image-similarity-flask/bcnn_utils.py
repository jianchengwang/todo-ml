import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib as mpl
import cv2
import time
import PIL
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from keras.initializers import glorot_normal
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)
from matplotlib import pyplot
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

"""
Calculates dot product of x[0] and x[1] for mini_batch 

Assuming both have same size and shape

@param
x -> [ (size_minibatch, total_pixels, size_filter), (size_minibatch, total_pixels, size_filter) ]

"""
def dot_product(x):

    return keras.backend.batch_dot(x[0], x[1], axes=[1,1]) / x[0].get_shape().as_list()[1] 

"""
Calculate signed square root

@param
x -> a tensor

"""

def signed_sqrt(x):

    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

"""
Calculate L2-norm

@param
x -> a tensor

"""

def L2_norm(x, axis=-1):

    return keras.backend.l2_normalize(x, axis=axis)

'''

    Take outputs of last layer of VGG and load it into Lambda layer which calculates outer product.
    
    Here both bi-linear branches have same shape.
    
    z -> output shape tuple
    x -> outpur og VGG tensor
    y -> copy of x as we modify x, we use x, y for outer product.
    
'''

def build_model(input_shape):
    tensor_input = keras.layers.Input(shape=input_shape)

#   load pre-trained model
    tensor_input = keras.layers.Input(shape=input_shape)
    

    
    model_detector = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input, 
                            include_top=False,
                            weights='imagenet')
    
    model_detector2 = keras.applications.vgg16.VGG16(
                            input_tensor=tensor_input, 
                            include_top=False,
                            weights='imagenet')
    
    
    model_detector2 = keras.models.Sequential(layers=model_detector2.layers)
  
    for i, layer in enumerate(model_detector2.layers):
        layer._name = layer.name  +"_second"

    model2 = keras.models.Model(inputs=[tensor_input], outputs = [model_detector2.layers[-1].output])
                       
    x = model_detector.layers[17].output
    z = model_detector.layers[17].output_shape
    y = model2.layers[17].output
    
    print(model_detector.summary())
    
    print(model2.summary())
#   rehape to (batch_size, total_pixels, filter_size)
    x = keras.layers.Reshape([z[1] * z[2] , z[-1]])(x)
        
    y = keras.layers.Reshape([z[1] * z[2] , z[-1]])(y)
    
#   outer products of x, y
    x = keras.layers.Lambda(dot_product)([x, y])
    
#   rehape to (batch_size, filter_size_vgg_last_layer*filter_vgg_last_layer)
    x = keras.layers.Reshape([z[-1]*z[-1]])(x)
        
#   signed_sqrt
    x = keras.layers.Lambda(signed_sqrt)(x)
        
#   L2_norm
    x = keras.layers.Lambda(L2_norm)(x)

#   FC-Layer

    initializer = tf.keras.initializers.GlorotNormal()
            
    x = keras.layers.Dense(units=258, 
                           kernel_regularizer=keras.regularizers.l2(0.0),
                           kernel_initializer=initializer)(x)

    tensor_prediction = keras.layers.Activation("softmax")(x)

    model_bilinear = keras.models.Model(inputs=[tensor_input],
                                        outputs=[tensor_prediction])
    
    
#   Freeze VGG layers
    for layer in model_detector.layers:
        layer.trainable = False
        

    sgd = keras.optimizers.SGD(lr=1.0, 
                               decay=0.0,
                               momentum=0.9)

    model_bilinear.compile(loss="categorical_crossentropy", 
                           optimizer=sgd,
                           metrics=["categorical_accuracy"])

    model_bilinear.summary()
    
    return model_bilinear