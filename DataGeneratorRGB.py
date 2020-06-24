import os
import cv2
import numpy as np
import tensorflow as tf
import albumentations as A


class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            images_dir,
            crop_size = 256,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.crop_size = crop_size
        # convert str names to class values on masks4

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data

        img = cv2.imread(self.images_fps[i])  # [:,:,0] #cv2.imread(self.images_fps[i], -1)

        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        img = img[:, :, ::-1]
        image = img[:, :int(img.shape[1] / 2), :]
        mask = img[:, int(img.shape[1] / 2):, :]
        ##
        # mask = mask / 255
        ##

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        X, Y = [], []
        for j in range(start, stop):
            X.append(self.dataset[j][0])
            Y.append(self.dataset[j][1])
        # transpose list of lists
        X = np.array(X)
        Y = np.array(Y)
        ######
        if len(X.shape) < 4:
            X = np.expand_dims(X, axis=3)
        return X, Y

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation(crop_size):
    train_transform = [
        # A.Normalize(always_apply=True, mean=(0.456),  std=(0.224)),
        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),

        # A.HorizontalFlip(p=0.5),  # 0.5
        # A.VerticalFlip(p=0.1),  # 0.1
        # A.ShiftScaleRotate(scale_limit=(-0.4, 0.0), rotate_limit=(-10, 10), shift_limit=(-0.1, 0.1), p=1,
        #                    border_mode=cv2.BORDER_CONSTANT, value=0),  # 0.7

        # A.PadIfNeeded(min_height=320, min_width=320, p=0.4, border_mode=0),  # 0.4

        # A.IAAAdditiveGaussianNoise(p=0.2),#0.2
        # A.IAAPerspective(p=0.6),  # 0.6

        # A.RandomBrightness(p=0.2),#0.2

        # A.OneOf(
        #     [
        #         # A.CLAHE(p=1),
        #          A.RandomBrightness(p=0.2),#0.2
        #         #A.RandomGamma(p=0.2),#0.2
        #     ],
        #     p=0.3,#0.3
        # ),

        # A.OneOf(
        #     [
        #         A.IAASharpen(p=0.8),#0.8
        #         A.Blur(blur_limit=3, p=0.8),#0.8
        #         A.MotionBlur(blur_limit=3, p=0.8),#0.8
        #     ],
        #     p=0.3,#0.3
        # ),

        # A.RandomContrast(limit=(-0.2, 0.2), p=0.2),#0.2
        # A.Lambda(mask=round_clip_0_1)

    ]
    return A.Compose(train_transform)

def get_validation_augmentation(crop_size):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [

        A.RandomCrop(height=crop_size, width=crop_size, always_apply=True),
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

