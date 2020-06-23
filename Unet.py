from tensorflow.keras.layers import Convolution2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import concatenate, ReLU
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from EfficientNet import EfficientNet

class Unet:

    def createDataset(self, imgs):
        dataset = []
        mult = 2 ** self.depth
        for img in imgs:
            H = int(img.shape[0] / mult) * mult
            W = int(img.shape[1] / mult) * mult

            dataset.append(img[:H, :W, :])

        return np.array(dataset)

    def check_input_size(self):
        if self.input_size[0] is not None:
            if self.input_size[0] % (2 * self.depth) != 0:
                print("ERROR: SIZE IS WRONG")
        if self.input_size[1] is not None:
            if self.input_size[1] % (2 * self.depth) != 0:
                print("ERROR: SIZE IS WRONG")

    def create_model(self):
        if not self.efnet:
            # MAX_POOLING -> AVARAGE_POOLING

            inputs = Input(self.input_size)
            conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

            conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)

            conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Convolution2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

            # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
            # conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
            # drop4 = Dropout(0.5)(conv4)
            # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
            #
            # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
            # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
            # drop5 = Dropout(0.5)(conv5)
            #
            # up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
            # merge6 = concatenate([drop4,up6], axis = 3)
            # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
            # conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

            # up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
            # merge7 = concatenate([conv3,up7], axis = 3)
            # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
            # conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

            up8 = UpSampling2D(size=(2, 2))(conv3)
            up8 = Convolution2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up8)

            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Convolution2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = UpSampling2D(size=(2, 2))(conv8)
            up9 = Convolution2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
            merge9 = concatenate([conv1, up9], axis=3)

            conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Convolution2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Convolution2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

            conv10 = Convolution2D(3, 1)(conv9)

            self.model = Model(inputs, conv10)

            self.depth = 2

        else:
            efnet = EfficientNet(input_size=self.input_size)

            X = efnet.outputs[0]

            for i in range(1, len(efnet.outputs)):
                f = efnet.outputs[i].shape[-1]

                X = UpSampling2D(size=(2, 2))(X)
                X = Convolution2D(filters=f, kernel_size=(2, 2), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(X)

                X = concatenate([efnet.outputs[i], X], axis=3)
                X = Convolution2D(filters=f, kernel_size=(3, 3), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(X)
                X = Convolution2D(filters=f, kernel_size=(3, 3), activation='relu', padding='same',
                                  kernel_initializer='he_normal')(X)

            X = Convolution2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same',
                              kernel_initializer='he_normal')(X)
            self.outputs = Convolution2D(filters=self.output_channels, kernel_size=(1, 1), activation='sigmoid',
                                         padding='same', kernel_initializer='he_normal')(X)

            self.model = Model(efnet.inputs, self.outputs)

            self.depth = efnet.depth

    def __init__(self, pretrained_model=None, input_size=(256, 256, 1), output_channels = 3, efnet = True, epochs = 10, batchsize = 8):

        self.EPOCHS = epochs
        self.BS = batchsize
        self.input_size = input_size
        self.efnet = efnet
        self.output_channels = output_channels

        self.create_model()

        self.check_input_size()

        if (pretrained_model is not None) and os.path.exists(pretrained_model):
            self.model.load_weights(pretrained_model)


    def compile(self):
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[sm.metrics.IOUScore(threshold=0.5)])

    def fit(self, x_train, y_train, x_valid, y_valid):
        callbacks = [
            # save model with best result (for loss we need min value)
            tf.keras.callbacks.ModelCheckpoint('./best_unet_model.h5', save_best_only=True, mode='min', save_weights_only=True),
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

