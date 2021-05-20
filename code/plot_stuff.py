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


def view_slices(data, pic_number):
    sample_ct = read_nii(data['ct_scan'][pic_number])
    sample_lung = read_nii(data['lung_mask'][pic_number])
    sample_infe = read_nii(data['infection_mask'][pic_number])
    sample_all = read_nii(data['lung_and_infection_mask'][pic_number])

    total_slices = sample_ct.shape[2]
    slice_idxs = [0, int(total_slices/2), -1]
    n_slices = len(slice_idxs)

    fig = plt.figure(figsize=(18, 15))

    for i in range(n_slices):
        plt.subplot(n_slices, 4, 4*i + 1)
        plt.imshow(sample_ct[..., slice_idxs[i]], cmap='bone')
        plt.title('Original Image')

        plt.subplot(n_slices, 4, 4*i + 2)
        plt.imshow(sample_ct[..., slice_idxs[i]], cmap='bone')
        plt.imshow(sample_lung[..., slice_idxs[i]],
                   alpha=0.5, cmap='nipy_spectral')
        plt.title('Lung Mask')

        plt.subplot(n_slices, 4, 4*i + 3)
        plt.imshow(sample_ct[..., slice_idxs[i]], cmap='bone')
        plt.imshow(sample_infe[..., slice_idxs[i]],
                   alpha=0.5, cmap='nipy_spectral')
        plt.title('Infection Mask')

        plt.subplot(n_slices, 4, 4*i + 4)
        plt.imshow(sample_ct[..., slice_idxs[i]], cmap='bone')
        plt.imshow(sample_all[..., slice_idxs[i]],
                   alpha=0.5, cmap='nipy_spectral')
        plt.title('Lung and Infection Mask')

    plt.savefig("slices.png")
    print("Slices saved in slices.png")


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


def plt_pics_from_side(lung_test, infect_test, pic_numbers, filename):
    fig = plt.figure(figsize=(18, 15))

    n_pics = len(pic_numbers)
    for i in range(n_pics):

        plt.subplot(n_pics, 3, 3*i+1)
        plt.imshow(lung_test[pic_numbers[i]], cmap='bone')
        plt.title('original lung')

        plt.subplot(n_pics, 3, 3*i+2)
        plt.imshow(lung_test[pic_numbers[i]], cmap='bone')
        plt.imshow(infect_test[pic_numbers[i]],
                   alpha=0.5, cmap="nipy_spectral")
        plt.title('original infection mask and lung')

        plt.subplot(n_pics, 3, 3*i+3)
        plt.imshow(infect_test[pic_numbers[i]],
                   alpha=0.5, cmap="nipy_spectral")
        plt.title('original infection mask only')

    plt.savefig(filename)
    print("Sideways images saved as {}".format(filename))


def plt_segmented(lung_test, infect_test, predicted, pic_numbers, filename):
    fig = plt.figure(figsize=(18, 15))

    n_pics = len(pic_numbers)
    for i in range(n_pics):

        plt.subplot(n_pics, 3, 3*i+1)
        plt.imshow(lung_test[pic_numbers[i]], cmap='bone')
        plt.title('original lung')

        plt.subplot(n_pics, 3, 3*i+2)
        plt.imshow(lung_test[pic_numbers[i]], cmap='bone')
        plt.imshow(infect_test[pic_numbers[i]],
                   alpha=0.5, cmap="nipy_spectral")
        plt.title('original infection mask')

        plt.subplot(n_pics, 3, 3*i+3)
        plt.imshow(infect_test[pic_numbers[i]],
                   alpha=0.5, cmap="nipy_spectral")
        # plt.imshow(lung_test[pic_number][..., 0], cmap='bone')
        # plt.imshow(predicted[pic_number][..., 0], alpha=0.5, cmap="nipy_spectral")
        plt.title('predicted infection mask')

    plt.savefig(filename)
    print("Segmented images saved as {}".format(filename))
