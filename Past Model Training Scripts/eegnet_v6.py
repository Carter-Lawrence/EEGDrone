import os
import numpy as np
import tensorflow as tf
import mne

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     SeparableConv2D, AveragePooling2D,
                                     Dropout, Dense, Flatten,
                                     BatchNormalization, Activation)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import SpatialDropout2D
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
def EEGNet_V6(chans, samples):

    inp = Input(shape=(chans, samples, 1))

    # -------- Temporal Convolution (frequency learning)
    x = Conv2D(32, (1, 64), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # -------- Spatial Convolution (channel interactions)
    x = DepthwiseConv2D((chans, 1), depth_multiplier=2,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.4)(x)

    # -------- Spectral abstraction
    x = SeparableConv2D(64, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(0.4)(x)

    # -------- Residual-like block (extra depth)
    res = SeparableConv2D(64, (1, 8), padding='same')(x)
    res = BatchNormalization()(res)
    res = Activation('elu')(res)
    x = Add()([x, res])

    # -------- Classifier
    x = Flatten()(x)
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    return Model(inp, out)
def EEGNet_V6(nb_classes, Chans, Samples, dropoutRate=0.20):
    inputs = Input(shape=(Chans, Samples, 1))

    x = Conv2D(16, (1, 64), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((Chans, 1),
                        depth_multiplier=2,
                        use_bias=False,
                        depthwise_constraint=max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = SpatialDropout2D(dropoutRate)(x)

    x = SeparableConv2D(32, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = SpatialDropout2D(dropoutRate)(x)

    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid',
                    kernel_constraint=max_norm(0.5))(x)

    return Model(inputs, outputs)