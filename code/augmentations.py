import cv2
import numpy as np
import nibabel as nib


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return(array)


def load_sideways(data, img_size, imgs, masks):
    print("\nStarting augmentation...")
    lungs = []
    infections = []
    antal = imgs.shape[0]
    for i in range(antal):
        ct = read_nii(imgs[i])
        infect = read_nii(masks[i])
        for ii in range(ct.shape[0]):
            lung_img = cv2.resize(ct[ii], dsize=(
                img_size, img_size), interpolation=cv2.INTER_AREA).astype('uint8')
            infec_img = cv2.resize(infect[ii], dsize=(
                img_size, img_size), interpolation=cv2.INTER_AREA).astype('uint8')
            lungs.append(lung_img)
            infections.append(infec_img)
    print("Augmentation done")
    return lungs, infections


def load_slices(data, img_size, imgs, masks):
    print("\nStarting augmentation...")
    lungs = []
    infections = []
    antal = imgs.shape[0]
    for i in range(antal):
        ct = read_nii(imgs[i])
        infect = read_nii(masks[i])
        for ii in range(ct.shape[2]):
            lung_img = cv2.resize(ct[..., ii], dsize=(
                img_size, img_size)).astype('uint8')
            infec_img = cv2.resize(infect[..., ii], dsize=(
                img_size, img_size)).astype('uint8')
            lungs.append(lung_img)
            infections.append(infec_img)
    print("Augmentation done")
    return lungs, infections
