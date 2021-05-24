# From our files
from unet import unet
from augmentations import *
from plot_stuff import *

# Required libraries
from sklearn.model_selection import train_test_split as split
import numpy
import pandas

allow_growth = True

# Parameters
TEST_SIZE = 0.1
EPOCHS = 5
STEPS_PER_EPOCH = 64
IMG_SIZE = 128
N_IMGS = 20

# Load data
metadata = pandas.read_csv('../input/covid19-ct-scans/metadata.csv')
metadata.head()

# View some samples (saved in "kolla.png") as my GUI is being a dick
view_slices(metadata, pic_number=0)

# Data augmentation
imgs = metadata['ct_scan']
masks = metadata['infection_mask']
scans, masks = load_slices(metadata, IMG_SIZE, imgs, masks, N_IMGS)
scans = numpy.array(scans)
masks = numpy.array(masks)

# For sideways: 6000
view_data([2, 60, 100, 150], scans, masks, "data.png")

# Splits
scan_train, scan_test, mask_train, mask_test = split(
    scans, masks, test_size=TEST_SIZE)

print("\nTrain set size: {}\nTest set size: {}".format(
    str(scan_train.shape[0]), str(scan_test.shape[0])))

# Create and compile U-net
model = unet((IMG_SIZE, IMG_SIZE, 1))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
# model.summary()
print("\nAaaand here we go with training the model...")
history = model.fit(scan_train, mask_train, epochs=EPOCHS,
                    validation_data=(scan_test, mask_test))
# steps_per_epoch=STEPS_PER_EPOCH
print("\nModel finished training")

# Save the results (taken from: https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object)
plt_acc(history)
plt_loss(history)

hist_df = pandas.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Segment new lungs with the trained U-net and plot them
segmented = model.predict(scan_test)
n_plots = 20
plot_many_segmented(n_plots, scan_test, mask_test, segmented)
