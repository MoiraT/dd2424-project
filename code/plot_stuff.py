import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return(array)


def view_samples(data, pic_number):
    sample_ct = read_nii(data.loc[pic_number, 'ct_scan'])
    sample_lung = read_nii(data.loc[pic_number, 'lung_mask'])
    sample_infe = read_nii(data.loc[pic_number, 'infection_mask'])
    sample_all = read_nii(data.loc[pic_number, 'lung_and_infection_mask'])

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

    plt.savefig("kolla.png")
    print("Sample picture saved as kolla.png")


def plt_acc(history):
    fig = plt.figure(figsize=(18, 15))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig("acc_vs_epoch.png")
    print("Plot of accuracy vs epochs saved as acc_vs_epoch.png")


def plt_loss(history):
    fig = plt.figure(figsize=(18, 15))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig("loss_vs_epoch.png")
    print("Plot of loss vs epochs saved as loss_vs_epoch.png")


def plt_segmented(lung_test, infect_test, predicted, pic_number):
    fig = plt.figure(figsize=(18, 15))

    plt.subplot(1, 3, 1)
    plt.imshow(lung_test[pic_number][..., 0], cmap='bone')
    plt.title('original lung')

    plt.subplot(1, 3, 2)
    plt.imshow(lung_test[pic_number][..., 0], cmap='bone')
    plt.imshow(infect_test[3][..., 0], alpha=0.5, cmap="nipy_spectral")
    plt.title('original infection mask')

    plt.subplot(1, 3, 3)
    plt.imshow(lung_test[pic_number][..., 0], cmap='bone')
    plt.imshow(predicted[pic_number][..., 0], alpha=0.5, cmap="nipy_spectral")
    plt.title('predicted infection mask')

    plt.savefig("result.png")
    print("Segmented images saved as result.png")
