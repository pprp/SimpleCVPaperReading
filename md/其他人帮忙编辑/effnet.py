# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import json
import cv2
import os
import h5py
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1
from tensorflow.keras.optimizers import Adam
# ignoring warnings
import warnings
warnings.simplefilter("ignore")

WORK_DIR = '../input/cassava-leaf-disease-classification'
print(os.listdir(WORK_DIR))
train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

TARGET_SIZE = 300
BATCH_SIZE = 30
train_labels.label = train_labels.label.astype('str')

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   preprocessing_function=None,
                                   #rotation_range = 20,
                                   #zoom_range = 0.2,
                                   #cval = 0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
#shear_range = 0.15,
#height_shift_range = 0.15,
#width_shift_range = 0.15,
#featurewise_center = True,
# featurewise_std_normalization = True)

train_generator = train_datagen.flow_from_dataframe(train_labels,
                                                    directory=os.path.join(
                                                        WORK_DIR, "train_images"),
                                                    subset="training",
                                                    x_col="image_id",
                                                    y_col="label",
                                                    target_size=(
                                                        TARGET_SIZE, TARGET_SIZE),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="sparse")


validation_datagen = ImageDataGenerator(validation_split=0.2)

validation_generator = validation_datagen.flow_from_dataframe(train_labels,
                                                              directory=os.path.join(
                                                                  WORK_DIR, "train_images"),
                                                              subset="validation",
                                                              x_col="image_id",
                                                              y_col="label",
                                                              target_size=(
                                                                  TARGET_SIZE, TARGET_SIZE),
                                                              batch_size=BATCH_SIZE,
                                                              class_mode="sparse")


def create_model():
    conv_base = EfficientNetB1(include_top=False, weights=None,
                               input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    model = conv_base.output
    model = layers.GlobalAveragePooling2D()(model)
    model = layers.Dense(5, activation="softmax")(model)
    model = models.Model(conv_base.input, model)

    model.compile(optimizer=Adam(lr=0.001),
                  loss="sparse_categorical_crossentropy",
                  metrics=["acc"])
    return model


model = create_model()
# model.summary()

model_save = ModelCheckpoint('EffNetB0_300_16_best_weights.h5',
                             save_best_only=True,
                             save_weights_only=False,
                             monitor='val_loss',
                             mode='min', verbose=1)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001,
                           patience=5, mode='min', verbose=1,
                           restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=2, min_delta=0.001,
                              mode='min', verbose=1)

STEPS_PER_EPOCH = len(train_labels)*0.8 / BATCH_SIZE
VALIDATION_STEPS = len(train_labels)*0.2 / BATCH_SIZE
EPOCHS = 10


history = model.fit(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr]
)

model.load_weights('EffNetB0_300_16_best_weights.h5')
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
preds = []

for image_id in ss.image_id:
    image = Image.open(os.path.join(WORK_DIR,  "test_images", image_id))
    image = image.resize((TARGET_SIZE, TARGET_SIZE))
    image = np.expand_dims(image, axis=0)
    print(model.predict(image))
    preds.append(np.argmax(model.predict(image)))

ss['label'] = preds
ss.to_csv('/kaggle/working/submission.csv', index=False)
