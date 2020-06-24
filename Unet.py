from tensorflow.keras.layers import Convolution2D, Input
from tensorflow.keras.layers import concatenate, ELU, Activation, Softmax
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
import numpy as np
import tensorflow as tf
import segmentation_models as sm

from CNN import CNN

class Unet(CNN):

    def createDataset(self, imgs):
        dataset = []
        mult = 2 ** self.depth
        for img in imgs:
            H = int(img.shape[0] / mult) * mult
            W = int(img.shape[1] / mult) * mult

            dataset.append(img[:H, :W, :])
        return np.array(dataset)

    def default_backbone(self):
        self.inputs = Input(self.input_size)

        X = self.inputs
        f = 64
        outputs = []

        for i in range(self.depth - 1):
            X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
            X = ELU()(X)
            X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
            X = ELU()(X)
            outputs.insert(0, X)
            X = AveragePooling2D(pool_size=(2, 2))(X)
            f *= 2

        X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
        X = ELU()(X)
        X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
        X = ELU()(X)

        outputs.insert(0, X)
        return outputs



    def create_model(self):
        super().create_model()

        X = self.outputs[0]

        for i in range(1, len(self.outputs)):
            f = self.outputs[i].shape[-1]

            X = UpSampling2D(size=(2, 2))(X)
            X = Convolution2D(filters=f, kernel_size=(2, 2), padding='same',kernel_initializer='he_normal')(X)
            X = ELU()(X)

            X = concatenate([self.outputs[i], X], axis=3)
            X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
            X = ELU()(X)

            X = Convolution2D(filters=f, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
            X = ELU()(X)

        X = Convolution2D(filters=2, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
        X = ELU()(X)

        if self.n_classes == 1:
            activation = 'sigmoid'
        else:
            activation = 'softmax'

        self.outputs = Convolution2D(filters=self.output_channels, kernel_size=(1, 1), activation=activation, padding='same', kernel_initializer='he_normal')(X)





    def __init__(self, pretrained_weights=None, input_size=(256, 256, 1), output_channels = 3, depth = 3, backbone=None, n_classes=1):
        self.depth = depth
        super().__init__(pretrained_weights, input_size, output_channels, n_classes, backbone)


