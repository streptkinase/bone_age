#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals
import sys
print(sys.path)
sys.path.append('/home/jupyter/.local/lib/python3.5/site-packages/')
print(sys.path)

# In[1]:


#!sudo lsblk


# In[2]:


#!sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb


# In[3]:


#!sudo mkdir -p /mnt/disks/mount_point


# In[1]:


#get_ipython().system(u'sudo mount -o discard,defaults /dev/sdb /mnt/disks/mount_point/')


# In[2]:


#get_ipython().system(u'sudo chmod a+w /mnt/disks/mount_point/')


# In[6]:


#!nvidia-smi


# In[7]:


#!pip install kaggle


# In[8]:


#!mkdir ~/.kaggle/


# In[9]:


#!cd ~


# In[10]:


#ls


# In[11]:


#!cp kaggle.json .kaggle/


# In[12]:


#!pip show


# In[13]:


#!/usr/local/bin/kaggle datasets download -d kmader/rsna-bone-age


# In[14]:


#ls


# In[15]:


#pwd


# In[16]:


#cd ..


# In[17]:


#ls


# In[18]:


#cd ..


# In[19]:


#!sudo find / -name "kaggle"


# In[20]:


#!/home/jupyter/.local/bin/kaggle datasets download -d kmader/rsna-bone-age


# In[21]:


#ls


# In[3]:


#cd /mnt/disks/mount_point/bone_age/


# In[4]:


#ls


# In[24]:


#mkdir bone_age


# In[25]:


#cd bone_age/


# In[26]:


#ls


# In[7]:


#!ps a


# In[27]:


#!unzip rsna-bone-age.zip


# In[28]:


#from PIL import Image
#import numpy as np

# 元となる画像の読み込み
#img = Image.open('boneage-training-dataset/boneage-training-dataset/10000.png')
#オリジナル画像の幅と高さを取得


# In[29]:


#img


# In[30]:


#img.getpixel((123, 99))


# In[4]:


import pandas as pd
age_df = pd.read_csv('boneage-training-dataset.csv')


# In[32]:


#age_df


# In[5]:


age_df['boneage_category'] = pd.cut(age_df['boneage'], 10)


# In[6]:


age_df['boneage'] = age_df['boneage'].astype(str)


# In[7]:


def f(x):
    return str(x) + '.png'


# In[8]:


age_df['path'] = age_df['id'].apply(f)


# In[9]:


test_df = pd.read_csv('boneage-test-dataset.csv')


# In[38]:


#test_df


# In[10]:


test_df['path'] = test_df['Case ID'].apply(f)


# In[11]:


from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(age_df, 
                                   test_size = 0.25, 
                                   random_state = 20,
                                   stratify = age_df['boneage_category'])
print('train', train_df.shape[0], 'validation', valid_df.shape[0])


# In[12]:



import os

import numpy as np

import matplotlib.pyplot as plt


# In[13]:


try:
  # %tensorflow_version only exist
  get_ipython().magic(u'tensorflow_version 2.x')
except Exception:
  pass
import tensorflow as tf
keras = tf.keras


# In[14]:


from keras.preprocessing.image import ImageDataGenerator
IMG_SIZE = (380, 380) 
idg = ImageDataGenerator(samplewise_center=False, 
                              samplewise_std_normalization=False, 
                              horizontal_flip = True, 
                              vertical_flip = True, 
                              height_shift_range = 0.15, 
                              width_shift_range = 0.15, 
                              rotation_range = 5, 
                              shear_range = 0.01,
                              fill_mode = 'nearest',
                              zoom_range=0.25,
                              rescale=1./255)


# In[15]:


train_gen = idg.flow_from_dataframe(train_df, 
                                    directory='/mnt/disks/mount_point/bone_age/boneage-training-dataset/boneage-training-dataset/',
                                   x_col='path',
                                   y_col='boneage',
                                   class_mode='sparse',
                                   batch_size=256,
                                   seed=10,
                                   target_size=(380, 380))


# In[16]:
test_idg = ImageDataGenerator(rescale=1./255)

valid_gen = test_idg.flow_from_dataframe(valid_df, 
                                    directory='/mnt/disks/mount_point/bone_age/boneage-training-dataset/boneage-training-dataset/',
                                   x_col='path',
                                   y_col='boneage',
                                   class_mode='sparse',
                                   batch_size=256,
                                   seed=10,
                                   shuffle=False,
                                   target_size=(380, 380))


# In[17]:


test_gen = idg.flow_from_dataframe(test_df, 
                                    directory='/mnt/disks/mount_point/bone_age/boneage-test-dataset/boneage-test-dataset/',
                                   x_col='path',
                                   class_mode=None,
                                   batch_size=100,
                                   seed=10,
                                   target_size=(380, 380))


# In[47]:


#!git clone https://github.com/surmenok/keras_lr_finder


# In[48]:


#ls


# In[49]:


#ls keras_lr_finder/keras_lr_finder/


# In[18]:


import sys


# In[19]:


sys.path.append('/mnt/disks/mount_point/bone_age/keras_lr_finder/keras_lr_finder/')


# In[20]:


sys.path.append('/home/jupyter/.local/lib/python3.5/site-packages/')


# In[21]:


#import lr_finder


# In[ ]:





# In[ ]:





# In[54]:


#!pip install git+https://github.com/qubvel/efficientnet


# In[55]:


#!nvidia-smi


# In[21]:


from efficientnet.tfkeras import EfficientNetB4


# In[57]:


#ls


# In[58]:


#!sudo find / -name "efficientnet"


# In[59]:


#ls /home/jupyter/.local/lib/python3.5/site-packages/efficientnet/


# In[22]:


base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))


# In[23]:


base_model.trainable = False


# In[62]:


#base_model.summary()


# In[63]:


#base_model.input


# In[24]:


x = base_model.output


# In[65]:


#x.shape


# In[25]:


global_average_layer = keras.layers.GlobalAveragePooling2D()


# In[26]:


averaged = global_average_layer(x)


# In[68]:


#averaged.shape


# In[27]:


prediction_layer = keras.layers.Dense(1, activation='linear')


# In[28]:


output = prediction_layer(averaged)


# In[71]:


#output.shape


# In[72]:


#from keras.models import Model


# In[73]:


#model_final = Model(input = base_model.input, output = output)


# In[29]:


final_model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])


# In[75]:


#final_model.summary()


# In[30]:


from keras.metrics import mean_absolute_error


# In[31]:


def mae_months(in_gt, in_pred):
    return mean_absolute_error(in_gt, in_pred)


# In[32]:


final_model.compile(optimizer=keras.optimizers.Adam(lr=0.8), 
                    loss = 'mse', metrics = [mae_months])


# In[79]:


#del lr_finder


# In[34]:


#lrfinder = lr_finder.LRFinder(final_model)


# In[81]:


#lrfinder.find_generator(train_gen, start_lr=0.0001, end_lr=1, epochs=5)


# In[82]:


#lrfinder.plot_loss(n_skip_beginning=20, n_skip_end=5)


# In[83]:


#lrfinder.find_generator(train_gen, start_lr=0.0001, end_lr=1, epochs=5)


# In[85]:


#lrfinder.plot_loss(n_skip_beginning=20, n_skip_end=1)


# In[35]:


#lrfinder.find_generator(train_gen, start_lr=0.1, end_lr=100, epochs=5)


# In[37]:


#lrfinder.plot_loss(n_skip_beginning=2, n_skip_end=1)


# In[33]:


from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="bone_age_best_b4_valmodified/b4"

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)


reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=50) # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]


# In[ ]:


history = final_model.fit_generator(train_gen, validation_data=valid_gen, epochs=200, callbacks=callbacks_list)
import json
def support_default(o):
        return float(o)
with open('history_b4_valmodified.json', 'w') as f:
    json.dump(history.history, f, default=support_default)


# In[38]:


#!nvidia-smi


# In[ ]:


#jupyter nbconvert --to script /home/jupyter/Untitled.ipynb

