from tensorflow.keras.layers import Convolution2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import Add, ReLU
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

class EfficientNet:
    def MBconvN(self, X, filter_size, N):
        f = X.shape[-1]
        X_skip = X
        # first block
        X = Convolution2D(filters=N * f, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)

        # second block
        X = DepthwiseConv2D(kernel_size=filter_size, padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)

        # third block
        X = Convolution2D(filters=f, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)

        # skip connection
        X = Add()([X, X_skip])
        X = ReLU()(X)

        return X

    def conv_block(self, X):
        X = Convolution2D(filters=32, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)

        return X

    def conv_1_by_1(self, X, filters):
        X = Convolution2D(filters=filters, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(X)
        X = BatchNormalization()(X)
        X = ReLU()(X)

        return X

    def ef_pipline(self, input_layer):
        # size: 224x224x3
        X = self.conv_block(input_layer)
        X_0 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        # size: 112x112x32
        X = self.MBconvN(X, filter_size=(3, 3), N=1)
        X = self.conv_1_by_1(X, 16)
        # size: 112x112x16
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X_1 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        # size: 56x56x16
        X = self.conv_1_by_1(X, 24)
        # size: 56x56x24
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_2 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        # size: 28x28x24
        X = self.conv_1_by_1(X, 40)
        # size: 28x28x40
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.conv_1_by_1(X, 80)
        # size: 28x28x80
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_3 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        # size: 14x14x80
        X = self.conv_1_by_1(X, 112)
        # size: 14x14x112
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X_4 = X
        X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
        # size: 7x7x112
        X = self.conv_1_by_1(X, 192)
        # size: 7x7x192
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(5, 5), N=6)
        X = self.MBconvN(X, filter_size=(3, 3), N=6)
        X = self.conv_1_by_1(X, 320)
        # size: 7x7x320

        self.depth = 5

        return [X, X_4, X_3, X_2, X_1, X_0]

    def createDataset(self, imgs):
        dataset = []
        for img in imgs:
            H = int(img.shape[0] / 32) * 32
            W = int(img.shape[1] / 32) * 32

            dataset.append(img[:H, :W, :])

        return np.array(dataset)

    def check_input_size(self):
        if self.input_size[0] is not None:
            if self.input_size[0] % (2 * self.depth) != 0:
                print("ERROR: SIZE IS WRONG")
        if self.input_size[1] is not None:
            if self.input_size[1] % (2 * self.depth) != 0:
                print("ERROR: SIZE IS WRONG")


    def __init__(self, pretrained_model=None, input_size=(224, 224, 1), epochs = 10, batchsize = 8):
        self.EPOCHS = epochs
        self.BS = batchsize
        self.input_size = input_size

        self.inputs = Input(input_size)
        self.outputs = self.ef_pipline(self.inputs)
        self.model = Model(self.inputs, self.outputs[0])

        self.check_input_size()

        # load pretrained model
        if (pretrained_model is not None) and os.path.exists(pretrained_model):
            self.model.load_weights(pretrained_model)


    def compile(self):
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics={'iou_score': sm.metrics.IOUScore(threshold=0.5)})

    def fit(self, x_train, y_train, x_valid, y_valid):
        callbacks = [
            # save model with best result (for loss we need min value)
            tf.keras.callbacks.ModelCheckpoint('./best_ef_model.h5', save_best_only=True, mode='min', save_weights_only=True),
            # reduce LR when metric hasn't changed
            tf.keras.callbacks.ReduceLROnPlateau(),
        ]

        self.history = self.model.fit(
            x=x_train, y=y_train,
            batch_size=self.BS,
            epochs=self.EPOCHS,
            callbacks=callbacks,
            validation_data=(x_valid, y_valid)
        )

    def show_graphs(self):
        acc = self.history.history['iou_score']
        val_acc = self.history.history['val_iou_score']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()

        plt.figure()
        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()