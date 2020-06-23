
#!pip install segmentation_models
import segmentation_models as sm

from tensorflow.keras.layers import Convolution2D, DepthwiseConv2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU, Add, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, UpSampling2D, AveragePooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from EfficientNet import EfficientNet


class PSPNet:
    def conv_block(self, X, filters, block):
        # resiudal block with dilated convolutions
        # add skip connection at last after doing convoluion operation to input X

        b = 'block_'+str(block)+'_' #name for block
        f1,f2,f3 = filters #filter sizes
        X_skip = X  #initiate layers
        # block_a
        #conv layer with f1 filters and 1x1 kernel, dilation_rate 1(normal), same = saves input shape, with weight initialize
        X = Convolution2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
                          padding='same',kernel_initializer='he_normal')(X)
        #norm
        X = BatchNormalization()(X)
        #activation
        X = LeakyReLU(alpha=0.2)(X)

        # block_b
        #~ 3x3 kernel, with dilation(over 1),
        X = Convolution2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
                          padding='same',kernel_initializer='he_normal')(X)
        #norm
        X = BatchNormalization()(X)
        #activation
        X = LeakyReLU(alpha=0.2)(X)

        # block_c
        #~like first
        X = Convolution2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
                          padding='same',kernel_initializer='he_normal')(X)
        #norm
        X = BatchNormalization()(X)

        # skip_conv, ~ with kernel 3x3
        X_skip = Convolution2D(filters=f3,kernel_size=(3,3),padding='same')(X_skip)
        #norm
        X_skip = BatchNormalization()(X_skip)
        # block_c + skip_conv
        X = Add(name=b+'add')([X,X_skip])
        #activation
        X = ReLU(name=b+'relu')(X)
        return X

    def base_feature_maps(self, input_layer):
        # base covolution module to get input image feature maps
        #from 3 channels to 256
        # block_1
        base = self.conv_block(input_layer,[32,32,64],'1')
        # block_2
        base = self.conv_block(base,[64,64,128],'2')
        # block_3
        base = self.conv_block(base,[128,128,256],'3')
        return base

    def pyramid_feature_maps(self, input_layer):
        if self.efnet:
            self.efnet = EfficientNet(input_size=self.input_size)
            self.inputs = self.efnet.inputs
            base = self.efnet.outputs[0]
            self.depth = self.efnet.depth
        else:
            base = self.base_feature_maps(input_layer)
            self.depth = 0

            # pyramid pooling module
        down_sizes_red = (base.shape[1], base.shape[2])
        down_sizes_yellow = (int(base.shape[1] / 2), int(base.shape[2] / 2))
        down_sizes_blue = (int(base.shape[1] / 3), int(base.shape[2] / 3))
        down_sizes_green = (int(base.shape[1] / 6), int(base.shape[2] / 6))


        up_sizes_red = (input_layer.shape[1], input_layer.shape[2])
        up_sizes_yellow = (int(input_layer.shape[1] / 2), int(input_layer.shape[2] / 2))
        up_sizes_blue = (int(input_layer.shape[1] / 3), int(input_layer.shape[2] / 3))
        up_sizes_green = (int(input_layer.shape[1] / 6), int(input_layer.shape[2] / 6))
        up_sizes_base = (int(input_layer.shape[1] / base.shape[1]), int(input_layer.shape[2] / base.shape[2]))
        #up_sizes_purple = (int(input_layer.shape[1] / 12), int(input_layer.shape[2] / 12))

        # red
        #from HxW to 1x1
        red = GlobalAveragePooling2D()(base)
        #convert to 4dim
        red = tf.keras.layers.Reshape((1, 1, red.shape[-1]))(red)
        #conv with 64 filters and kernel 1x1
        red = Convolution2D(filters=64,kernel_size=(1,1))(red)
        #upsampling to original size
        red = UpSampling2D(size=up_sizes_red, interpolation='bilinear')(red)

        # yellow
        yellow = AveragePooling2D(pool_size = down_sizes_yellow, padding = 'valid')(base)
        # from HxW to H/2xW/2
        ##yellow = AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
        #conv with 64 filters and kernel 1x1
        yellow = Convolution2D(filters=64,kernel_size=(1,1))(yellow)
        #upsampling to original size
        yellow = UpSampling2D(size = up_sizes_yellow,interpolation='bilinear')(yellow)

        # blue
        blue = AveragePooling2D(pool_size=down_sizes_blue, padding = 'valid')(base)
        # from HxW to H/4xW/4
        ## blue = AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
        #conv with 64 filters and kernel 1x1
        blue = Convolution2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)
        #upsampling to original size
        blue = UpSampling2D(size=up_sizes_blue,interpolation='bilinear')(blue)

        # green
        green = AveragePooling2D(pool_size=down_sizes_green, padding = 'valid')(base)
        # from HxW to H/8xW/8
        ## green = AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
        #conv with 64 filters and kernel 1x1
        green = Convolution2D(filters=64,kernel_size=(1,1))(green)
        #upsampling to original size
        green = UpSampling2D(size=up_sizes_green,interpolation='bilinear')(green)

        base = UpSampling2D(size=up_sizes_base,interpolation='bilinear')(base)
        # # purple
        # purple = AveragePooling2D(pool_size=up_sizes_purple, padding = 'valid', name='purple_pool')(base)
        # # from HxW to H/16xW/16
        # ## purple = AveragePooling2D(pool_size=(16,16),name='purple_pool')(base)
        # #conv with 64 filters and kernel 1x1
        # purple = Convolution2D(filters=64,kernel_size=(1,1),name='purple_1_by_1')(purple)
        # #upsampling to original size
        # purple = UpSampling2D(size=up_sizes_purple,interpolation='bilinear',name='purple_upsampling')(purple)


        self.devide = 6 #-> divide count
        # base + red + yellow + blue + green
        return tf.keras.layers.concatenate([base, red, yellow, blue, green])
        #return tf.keras.layers.concatenate([base,red,yellow,blue,green, purple])

    def last_conv_module(self, input_layer):
        #make pyra maps
        X = self.pyramid_feature_maps(input_layer)
        #conv with 3 filters and kernel 3x3
        X = Convolution2D(filters=3,kernel_size=3,padding='same')(X)
        #norm
        X = BatchNormalization()(X)
        #activation
        X = Activation('sigmoid')(X)
        # X = tf.keras.layers.Flatten(name='last_conv_flatten')(X)
        return X

    def createDataset(self, imgs):
        dataset = []
        total = len(imgs)
        cur = 0
        mult = ((2 ** self.depth) * self.devide)

        for img in imgs:
            H = int(img.shape[0] / mult) * mult
            W = int(img.shape[1] / mult) * mult

            dataset.append(img[:H, :W, :])
            cur += 1
            print(f'{round(cur * 100 / total)}%', end='\r')

        print()
        return np.array(dataset)

    def check_input_size(self):
        if self.input_size[0] is not None:
            if self.input_size[0] % ((2 ** self.depth) * self.devide) != 0:
                print(f"PSP ERROR: SIZE IS WRONG({self.input_size[0]} % {((2 ** self.depth) * self.devide)} != 0)")
        if self.input_size[1] is not None:
            if self.input_size[1] % ((2 ** self.depth) * self.devide) != 0:
                print(f"PSP ERROR: SIZE IS WRONG({self.input_size[1]} % {((2 ** self.depth) * self.devide)} != 0)")

    def __init__(self, pretrained_model = None, input_size = (192,192, 1), efnet=True):
        self.efnet = efnet
        self.input_size = input_size

        self.inputs = Input(input_size)
        self.outputs = self.last_conv_module(self.inputs)
        self.model = Model(self.inputs, self.outputs)

        self.check_input_size()

        #load pretrained model
        if (pretrained_model is not None) and os.path.exists(pretrained_model):
            self.model.load_weights(pretrained_model)

    def compile(self):
        self.model.summary()
        self.model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=[sm.metrics.IOUScore(threshold=0.5)])

    def fit(self, x_train = None, y_train = None, valid_data = None, BS=None, EPOCHS=1):
        callbacks = [
            # save model with best result (for loss we need min value)
            tf.keras.callbacks.ModelCheckpoint('./best_psp_model.h5', save_best_only=True, mode='min', save_weights_only=True),
            # reduce LR when metric hasn't changed
            tf.keras.callbacks.ReduceLROnPlateau(),
        ]

        self.history = self.model.fit(
            x=x_train, y=y_train,
            batch_size=BS,
            epochs= EPOCHS,
            callbacks=callbacks,
            validation_data=valid_data
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