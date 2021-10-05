# USAGE
# python train_succulents_detector.py --dataset dataset

# import the necessary packages

TF_CPP_MIN_LOG_LEVEL=2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


        
        
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default = 'dataset',
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="succulents2.model",
	help="path to output succulents detector model")
args = vars(ap.parse_args())

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-5
EPOCHS = 2
BS = 12

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# load the input image (224x224) and preprocess it
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)




# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding on the labels
lb = LabelEncoder()
labels = lb.fit_transform(labels)
#labels = lb.transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing


(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
del data
del labels
del imagePaths
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
 	horizontal_flip=True,
	fill_mode="nearest")

#modelo artigo


# base = tf.keras.applications.VGG19(input_shape=(224, 224, 3),
#                                   include_top = False,
#                                   weights ='imagenet',
#                                   pooling = 'max')

base = tf.keras.applications.MobileNetV2(weights="imagenet",
                                              include_top=False,
                                              input_tensor=Input(shape=(224, 224, 3)))


base.summary()
for layer in base.layers:
    layer.trainable = False
cnn = tf.keras.models.Sequential()

cnn.add(base)

# cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[224, 224, 3]))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# cnn.add(tf.keras.layers.Dropout(.25))


# cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
# cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# cnn.add(tf.keras.layers.Dropout(.25))

# cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# cnn.add(tf.keras.layers.Dropout(.25))

# cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", activation="relu"))
# cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# cnn.add(tf.keras.layers.Dropout(.25))
cnn.add(tf.keras.layers.Flatten(input_shape=base.output_shape[1:]))

cnn.add(tf.keras.layers.Dense(units=128*2, activation='relu'))
cnn.add(tf.keras.layers.Dropout(.5))
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dropout(.5))

cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))


cnn.summary()

# compile our model
print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

opt = tf.keras.optimizers.SGD(lr=INIT_LR, momentum=0.9)
cnn.compile(loss="binary_crossentropy", optimizer=opt,	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
callback = EarlyStopping(monitor='loss', patience=3)

H = cnn.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS, 
    callbacks = [callback]#,
    # workers =4,
    # use_multiprocessing= True
    )



# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = cnn.predict(testX, batch_size=BS)
predIdxs2 =cnn.predict(testX)



# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)



# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
df_cm = pd.DataFrame(cm, range(10), range(10))
df_cm.index = lb.classes_
df_cm.columns= lb.classes_
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.show()
plt.clf()

# serialize the model to disk
print("[INFO] saving succulents detector model...")
cnn.save(args["model"], save_format="h5")

H= load_model(args["model"])

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.show()
plt.clf()



plt.style.use("ggplot")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.show()

# plt.savefig(args["plot"])
