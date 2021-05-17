#from unet import unet
from preprocess import *

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
#import tensorflow as tf

# crop_images(10)
#model = unet((256, 256, 1))

data = pd.read_csv('../input/covid19-ct-scans/metadata.csv')
data.head()


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return(array)


sample_ct = read_nii(data.loc[1, 'ct_scan'])
sample_lung = read_nii(data.loc[1, 'lung_mask'])
sample_infe = read_nii(data.loc[1, 'infection_mask'])
sample_all = read_nii(data.loc[1, 'lung_and_infection_mask'])

fig = plt.figure(figsize=(18, 15))
plt.subplot(1, 4, 1)
plt.imshow(sample_ct[..., 150], cmap='bone')
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(sample_ct[..., 150], cmap='bone')
plt.imshow(sample_lung[..., 150], alpha=0.5, cmap='nipy_spectral')
plt.title('Lung Mask')

plt.subplot(1, 4, 3)
plt.imshow(sample_ct[..., 150], cmap='bone')
plt.imshow(sample_infe[..., 150], alpha=0.5, cmap='nipy_spectral')
plt.title('Infection Mask')

plt.subplot(1, 4, 4)
plt.imshow(sample_ct[..., 150], cmap='bone')
plt.imshow(sample_all[..., 150], alpha=0.5, cmap='nipy_spectral')
plt.title('Lung and Infection Mask')
