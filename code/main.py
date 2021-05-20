# From our files
from unet import unet
# from plot_stuff import read_nii, plt_acc, plt_loss, view_samples, plt_segmented
from augmentations import load_sideways, load_slices
from plot_stuff import *

# Required libraries
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Parameters
TEST_SIZE = 0.1
EPOCHS = 1
STEPS_PER_EPOCH = 64
img_size = 128

# Load data
data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')
data.head()

# View some samples (saved in "kolla.png") as my GUI is being a dick
view_slices(data, pic_number=2)


# Data augmentation
imgs = data['ct_scan']
masks = data['infection_mask']
lungs, infections = load_sideways(data, img_size, imgs, masks)

lungs = np.array(lungs)
infections = np.array(infections)
print("\nLung array shape: {}, infection array shape: {}".format(
    lungs.shape, infections.shape))

plt_pics_from_side(lungs, infections,
                   [100, 550, 1000, 6000], "sideways_pics.png")
# For sideways: 6000

# Splits
lung_train, lung_test, infect_train, infect_test = train_test_split(
    lungs, infections, test_size=TEST_SIZE)

print("\nTrain set size: {}\nTest set size: {}".format(
    str(lung_train.shape[0]), str(lung_test.shape[0])))

# Create and compile U-net
model = unet((img_size, img_size, 1))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# model.summary()
print("\nAaaand here we go with training the model...")
history = model.fit(lung_train, infect_train, epochs=EPOCHS,
                    validation_data=(lung_test, infect_test))
# steps_per_epoch=STEPS_PER_EPOCH
print("\nModel finished training")

# Plot results
plt_acc(history)
plt_loss(history)

# Segment new lungs with the trained U-net
predicted = model.predict(lung_test)
plt_segmented(lung_test, infect_test, predicted,
              [0, 100, 550, 1000], "result.png")
