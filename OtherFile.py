import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize

###################################################################################################
# Define Variables¶

image_shape = (768, 768)
IMG_CHANNELS = 3
TARGET_WIDTH = 128
TARGET_HEIGHT = 128

no_mask = np.zeros(image_shape[0]*image_shape[1], dtype=np.uint8)

###################################################################################################
# Returns the run-length encoding (RLE) of the input image as a string formatted sequence.

def rle_encode(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    rle = ' '.join(str(x) for x in runs)
    return rle


###################################################################################################
# Decodes the run-length encoded (RLE) mask back into its original 2D binary mask representation.

def rle_decode(mask_rle, shape=image_shape):
    if pd.isnull(mask_rle):
        img = no_mask
        return img.reshape(shape).T
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype = int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype = np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


###################################################################################################
# Reads and preprocesses an image

def get_image(image_name):
    img = imread('data/train_v2/' + image_name)[:, :, :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode = 'constant', preserve_range = True)
    return img


###################################################################################################
# Decodes and preprocesses a run-length encoded (RLE) mask.

def get_mask(code):
    img = rle_decode(code)
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT, 1), mode = 'constant', preserve_range = True)
    return img

###################################################################################################
# Reads and preprocesses test image

def get_test_image(image_name):
    img = imread('../input/test_v2/' + image_name)[:, :, :IMG_CHANNELS]
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)
    return img
###################################################################################################
# Creates a generator that yields batches of preprocessed test images.

def create_test_generator(precess_batch_size, sub_df):
    while True:
        for k, ix in sub_df.groupby(np.arange(sub_df.shape[0]) // precess_batch_size):
            imgs = []
            for index, row in ix.iterrows():
                original_img = get_test_image(row.ImageId) / 255.0
                imgs.append(original_img)

            imgs = np.array(imgs)
            yield imgs

###################################################################################################
# Creates a generator that yields batches of preprocessed images and corresponding masks.

def create_image_generator(precess_batch_size, data_df):
    while True:
        for k, group_df in data_df.groupby(np.arange(data_df.shape[0]) // precess_batch_size):
            imgs = []
            labels = []
            for index, row in group_df.iterrows():
                # images
                original_img = get_image(row.ImageId) / 255.0
                # masks
                mask = get_mask(row.EncodedPixels) / 255.0

                imgs.append(original_img)
                labels.append(mask)

            imgs = np.array(imgs)
            labels = np.array(labels)
            yield imgs, labels