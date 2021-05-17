# From our files
from unet import unet
from plot_stuff import read_nii, plt_acc, plt_loss, view_samples, plt_segmented

# Required libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2

# Or else it'll be a warning bonanza
#TF_CPP_MIN_LOG_LEVEL = 2

# Parameters
TEST_SIZE = 0.1
EPOCHS = 2
STEPS_PER_EPOCH = 1
img_size = 128

# Load data
data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')
data.head()

# View some samples (saved in "kolla.png") as my GUI is being a dick
view_samples(data)

# Data augmentation
print("Starting augmentation")
lungs = []
infections = []
antal = 1  # len(data)
for i in range(antal):
    ct = read_nii(data['ct_scan'][i])
    infect = read_nii(data['infection_mask'][i])
    for ii in range(ct.shape[0]):
        lung_img = cv2.resize(ct[ii], dsize=(
            img_size, img_size), interpolation=cv2.INTER_AREA).astype('uint8')
        infec_img = cv2.resize(infect[ii], dsize=(
            img_size, img_size), interpolation=cv2.INTER_AREA).astype('uint8')
        lungs.append(lung_img[..., np.newaxis])
        infections.append(infec_img[..., np.newaxis])

print("Augmentation done")

lungs = np.array(lungs)
infections = np.array(infections)
print("Lung array shape: {}, infection array shape: {}".format(
    lungs.shape, infections.shape))

# Split
lung_train, lung_test, infect_train, infect_test = train_test_split(
    lungs, infections, test_size=TEST_SIZE)

print("Train set size: {}\nTest set size: {}".format(
    str(lung_train.shape[0]), str(lung_test.shape[0])))

# Create and compile U-net
model = unet((img_size, img_size, 1))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# model.summary()

history = model.fit(lung_train, infect_train, EPOCHS, STEPS_PER_EPOCH,
                    validation_data=(lung_test, infect_test))

# Plot results
plt_acc(history)
plt_loss(history)

# Segment new lungs with the trained U-net
predicted = model.predict(lung_test)
plt_segmented(lung_test, infect_test, predicted, 3)
