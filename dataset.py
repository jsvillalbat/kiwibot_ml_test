from PIL import Image
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

SIZE_X = 256
SIZE_Y = 256
n_classes=6

train_data_dir = "/tmp/dataset/train/"
test_data_dir = "/tmp/dataset/test/"

def create_example(data_frame, index):
  path_image=data_frame['image_path'][index]
  path = os.path.join(train_data_dir,path_image)
  mask_path=data_frame['label_path'][index]
  mpath = os.path.join(train_data_dir,mask_path)
  # image
  image = Image.open(path)
  image = np.array(image)
  image = tf.image.resize(image,[SIZE_X,SIZE_Y]).numpy()
  #mask
  mask = Image.open(mpath)
  mask = np.array(mask) 
  mask = mask[:,:,np.newaxis]
  mask = tf.image.resize(mask,[SIZE_X,SIZE_Y],method='nearest').numpy()
  mask = np.squeeze(mask)
  #label img
  label_img = data_frame['image_lvl_label'][index]
  return image, label_img, mask

def generate_data(data_frame,batch_size=8):
  num_examples=len(data_frame)
  labelencoder = LabelEncoder()

  while True:
    image_batch=np.zeros((batch_size,SIZE_X,SIZE_Y,3))
    label_batch=np.zeros((batch_size,))
    mask_batch=np.zeros((batch_size,SIZE_X,SIZE_Y))

    for i in range(0,batch_size):
      index = np.random.randint(0,num_examples)
      image, label, mask = create_example(data_frame,index)

      image_batch[i] = image
      label_batch[i] = label
      mask_batch[i] = mask
    
    #preprocess
    n, h, w = mask_batch.shape  
    mask_dataset_reshaped = mask_batch.reshape(-1,1)
    mask_dataset_reshaped_encoded = labelencoder.fit_transform(mask_dataset_reshaped)
    mask_dataset_encoded = mask_dataset_reshaped_encoded.reshape(n, h, w)
    mask_dataset_encoded = mask_dataset_encoded[:,:,:,np.newaxis]
    mask_dataset_encoded = to_categorical(mask_dataset_encoded, num_classes=n_classes)

    yield image_batch/255., [label_batch, mask_dataset_encoded]