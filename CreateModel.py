import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from OtherFile import create_image_generator
import os
from Model import Create_Model

###################################################################################################
# Define Variables¶

IMG_WIDTH = 768
IMG_HEIGHT = 768
IMG_CHANNELS = 3
epochs = 1
batch_size = 10
FAST_RUN = False  # use for development only
FAST_PREDICTION = False  # use for development only
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

###################################################################################################
# Read and Split Data

df = pd.read_csv("data/train_ship_segmentations_v2.csv")

train_df, validate_df = train_test_split(df)

###################################################################################################
# Create generators

train_generator = create_image_generator(batch_size, train_df)
validate_generator = create_image_generator(batch_size, validate_df)


###################################################################################################
# create a callback to save model checkpoint

early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
train_steps = np.ceil(float(train_df.shape[0]) / float(batch_size)).astype(int)
validate_steps = np.ceil(float(validate_df.shape[0]) / float(batch_size)).astype(int)
model = Create_Model()

###################################################################################################
# Fit model

history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validate_generator,
    validation_steps=validate_steps,
    epochs=epochs,
    callbacks = [early_stop]
)

###################################################################################################
# Save weight

model.save_weight('Checkpoint/ShipDetection_model.h5')