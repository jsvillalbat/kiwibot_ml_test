import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf

def visualize(image,label,mask):
  print(label)
  fig, (ax1,ax2) = plt.subplots(1,2)
  ax1.imshow(image)
  im = ax2.imshow(mask)
  values = np.unique(mask)
  colors = [ im.cmap(im.norm(value)) for value in values]
  patches = [ mpatches.Patch(color=colors[i], label="Seg: {l}".format(l=values[i]) ) for i in range(len(values)) ]
  ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
  plt.show()

class Logger(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    img_accuracy = logs.get('img_accuracy')
    seg_accuracy = logs.get('seg_accuracy')
    val_img_accuracy = logs.get('val_img_accuracy')
    val_seg_accuracy = logs.get('val_seg_accuracy')
    print('='*30, epoch + 1, '='*30)
    print(f'img_accuracy: {img_accuracy:.3f}, seg_accuracy: {seg_accuracy:.3f}')
    print(f'val_img_accuracy: {val_img_accuracy:.3f}, val_seg_accuracy: {val_seg_accuracy:.3f}')