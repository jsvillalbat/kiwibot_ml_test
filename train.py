from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from model import build_unet
from dataset import generate_data
from utils import Logger
import pandas as pd
import numpy as np

IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

save_model_dir= "/gdrive/My Drive/kiwibotml/saved_models/simpleunet_50epochs_total.hdf5"

train_data_dir = "/tmp/dataset/train/"
test_data_dir = "/tmp/dataset/test/"
data_csv_train_dir = "/tmp/dataset/train/data.csv"
data_csv_test_dir = "/tmp/dataset/test/data.csv"
data_train = pd.read_csv(data_csv_train_dir)

model = build_unet(input_shape, n_classes=6)
model.compile(
    optimizer='adam', 
    loss={
        'img':'sparse_categorical_crossentropy',
        'seg': 'categorical_crossentropy'
    }, 
    metrics=['accuracy'])

print(model.summary())

#Load data
data_labeled = data_train[:2000]
train_gen_data = data_labeled[:1500]
validation_gen_data = data_labeled[1500:]
validation_gen_data.set_index(np.arange(0,len(validation_gen_data)),inplace=True)

train_gen = generate_data(train_gen_data)
val_gen = generate_data(validation_gen_data)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch=200,
    validation_steps = 100,
    epochs=50, 
    callbacks=[
        Logger(),
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
    ],
    verbose=False)

model.save(save_model_dir)