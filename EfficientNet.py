from tensorflow.keras.layers import Convolution2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import Add, ELU
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from CNN import CNN

class EfficientNet(CNN):

    def MBconvN(self, X, filter_size, N):
        f = X.shape[-1]
        X_skip = X
        X = Convolution2D(filters=N * f, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ELU()(X)

        X = DepthwiseConv2D(kernel_size=filter_size, padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ELU()(X)

        X = Convolution2D(filters=f, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)

        X = Add()([X, X_skip])
        X = ELU()(X)

        return X

    def conv_block(self, X):
        X = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ELU()(X)

        return X

    def conv_1_by_1(self, X, filters):
        X = Convolution2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ELU()(X)

        return X

    def default_backbone(self):
        self.inputs = Input(self.input_size)

        X = self.conv_block(self.inputs)
        X_0 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X = self.MBconvN(X, filter_size=(3, 3), N=1)
        X = self.conv_1_by_1(X, 16)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X_1 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X = self.conv_1_by_1(X, 24)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_2 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X = self.conv_1_by_1(X, 40)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.conv_1_by_1(X, 80)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_3 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X = self.conv_1_by_1(X, 112)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_4 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

        X = self.conv_1_by_1(X, 192)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.conv_1_by_1(X, 320)

        self.depth = 5

        return [X, X_4, X_3, X_2, X_1, X_0]

    def create_model(self):
        super().create_model()


    def createDataset(self, imgs):
        dataset = []
        for img in imgs:
            H = int(img.shape[0] / 32) * 32
            W = int(img.shape[1] / 32) * 32

            dataset.append(img[:H, :W, :])

        return np.array(dataset)


    def __init__(self, pretrained_weights=None, input_size=(384, 384, 1)):
        super().__init__(pretrained_weights, input_size)