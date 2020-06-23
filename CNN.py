from abc import ABC, abstractmethod

from tensorflow.keras.layers import Convolution2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import matplotlib.pyplot as plt

class CNN(ABC):

    # def inputs_getter(self):
    #     return self._inputs
    # def inputs_setter(self, val):
    #     self._inputs = val
    # inputs = property(inputs_getter, inputs_setter)
    #
    # def outputs_getter(self):
    #     return self._outputs
    # def outputs_setter(self, val):
    #     self._outputs = val
    # outputs = property(outputs_getter, outputs_setter)

    @abstractmethod
    def default_backbone(self):
        pass


    @abstractmethod
    def create_model(self):
        if self.backbone is None:
            self.outputs = self.default_backbone()
        else:
            self.outputs = self.backbone.outputs
        # last layer -> self.outputs
        pass



    def __init__(self, pretrained_weights = None, input_size = (None, None, 3), output_channels = 1, n_classes = 10, backbone=None):
        self.output_channels = output_channels
        self.n_classes = n_classes
        self.backbone = backbone
        self.input_size = input_size

        # creation of model
        self.create_model()
        self.model = Model(self.inputs, self.outputs)

        if (pretrained_weights is not None) and os.path.exists(pretrained_weights):
            self.model.load_weights(pretrained_weights)


    def compile(self, optim=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.MeanIoU(num_classes=10)]):
        self.model.summary()
        self.model.compile(optimizer=optim, loss=loss, metrics=metrics)


    def fit(self, x_train=None, y_train=None, valid_data=None, BS=None, EPOCHS=1, path_to_save=None):
        if path_to_save is not None:
            callbacks = [
                # save model with best result (for loss we need min value)
                tf.keras.callbacks.ModelCheckpoint(path_to_save, save_best_only=True, mode='min', save_weights_only=True),
                # reduce LR when metric hasn't changed
                tf.keras.callbacks.ReduceLROnPlateau(),
            ]
        else:
            callbacks = None

        self.history = self.model.fit(
            x=x_train, y=y_train,
            batch_size=BS,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=valid_data
        )


    def show_graphs(self):

        metrics = self.model.metrics_names

        for met in metrics:
            plt.figure()

            met1 = self.history.history[met]
            met2 = self.history.history['val_' + met]

            epochs = range(1, len(met1) + 1)

            plt.plot(epochs, met1, 'b', label=f'Training {met}')
            plt.plot(epochs, met2, 'r', label=f'Validation {met}')
            plt.title(f'Training and Validation {met}')
            plt.legend()

        plt.show()

