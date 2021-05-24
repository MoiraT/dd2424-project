
import nibabel
import numpy
import matplotlib.pyplot as plt


def unpack(folder):
    ct_scan = nibabel.load(folder)
    imgs = numpy.rot90(numpy.array(ct_scan.get_fdata()))
    return(imgs)


def view_slices(data, pic_number):
    scans = unpack(data['ct_scan'][pic_number])
    infect = unpack(data['infection_mask'][pic_number])

    total_slices = scans.shape[2]

    slice_idxs = [int(total_slices*0.2), int(total_slices/2),
                  int(total_slices*0.8)]
    n_slices = len(slice_idxs)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        "First, middlest and last slice of the same lung scan", fontsize="30")

    for i in range(n_slices):
        plt.subplot(n_slices, 2, 2*i + 1)
        plt.axis("off")
        plt.imshow(scans[..., slice_idxs[i]])
        plt.title('CT scan', fontsize=25)

        plt.subplot(n_slices, 2, 2*i + 2)
        plt.axis("off")
        plt.imshow(scans[..., slice_idxs[i]])
        plt.imshow(infect[..., slice_idxs[i]],
                   cmap='nipy_spectral', alpha=0.6)
        plt.title('Covid infection mask over CT scan', fontsize=25)

    plt.savefig("../plots/slices.png")
    print("Slices saved in slices.png")


def plt_acc(history):
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig("../plots/acc_vs_epoch.png")
    print("Plot of accuracy vs epochs saved as acc_vs_epoch.png")


def plt_loss(history):
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.savefig("../plots/loss_vs_epoch.png")
    print("Plot of loss vs epochs saved as loss_vs_epoch.png")


def view_data(pic_idxs, scans, infect, filename):

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        "Four scans from the set and their infection masks", fontsize="30")
    n_pics = len(pic_idxs)
    for i in range(n_pics):

        plt.subplot(n_pics, 3, 3*i+1)
        plt.axis("off")
        plt.imshow(scans[pic_idxs[i]], cmap="bone")
        plt.title('CT scan', fontsize=25)

        plt.subplot(n_pics, 3, 3*i+2)
        plt.axis("off")
        plt.imshow(scans[pic_idxs[i]], cmap="bone")
        plt.imshow(infect[pic_idxs[i]],
                   alpha=0.5, cmap='nipy_spectral')
        plt.title('Ground truth infection mask', fontsize=25)

        plt.subplot(n_pics, 3, 3*i+3)
        plt.axis("off")
        plt.imshow(infect[pic_idxs[i]], alpha=0.5, cmap='nipy_spectral')
        plt.title('Infection mask only', fontsize=25)

    plt.savefig("../plots/" + filename)
    print("Images saved as data.png")


def plt_segmented(lung_test, infect_test, predicted, pic_numbers, filename):
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        "The original infection masks and the predicted one", fontsize="30")

    n_pics = len(pic_numbers)
    for i in range(n_pics):

        plt.subplot(n_pics, 3, 3*i+1)
        plt.axis("off")
        plt.imshow(lung_test[pic_numbers[i]], cmap="bone")
        plt.title('CT scan', fontsize=25)

        plt.subplot(n_pics, 3, 3*i+2)
        plt.axis("off")
        plt.imshow(lung_test[pic_numbers[i]], cmap="bone")
        plt.imshow(infect_test[pic_numbers[i]],
                   alpha=0.5, cmap='nipy_spectral')
        plt.title('Ground truth infection mask', fontsize=25)

        plt.subplot(n_pics, 3, 3*i+3)
        plt.axis("off")
        plt.imshow(lung_test[pic_numbers[i]], cmap="bone")
        plt.imshow(predicted[pic_numbers[i]], alpha=0.5, cmap='nipy_spectral')
        plt.title('Segmented infection mask', fontsize=25)

    plt.savefig("../plots/" + filename)
    print("Segmented images saved as {}".format(filename))


def plot_many_segmented(n_plots, scan_test, mask_test, segmented):
    total_imgs = scan_test.shape[0]
    blocks = int(total_imgs/n_plots)
    for i in range(n_plots):
        factor = n_plots * i
        plt_segmented(scan_test, mask_test, segmented,
                      [factor, factor+1, factor+2, factor+3], "result_slice_" + str(i+1) + ".png")
