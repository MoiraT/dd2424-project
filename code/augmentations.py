import nibabel
import cv2
import numpy


def unpack(folder):
    ct_scan = nibabel.load(folder)
    imgs = numpy.rot90(numpy.array(ct_scan.get_fdata()))
    return(imgs)


def load_sideways(data, size, imgs, masks, n_images):
    print("\nStarting augmentation...")
    img_array = []
    mask_array = []
    new_dim = (size, size)
    for i in range(n_images):
        img = unpack(imgs[i])
        mask = unpack(masks[i])
        dim = img.shape[0]
        for j in range(dim):
            new_img = cv2.resize(
                img[j], new_dim, interpolation=cv2.INTER_AREA).astype('uint8')
            new_mask = cv2.resize(
                mask[j], new_dim, interpolation=cv2.INTER_AREA).astype('uint8')
            img_array.append(new_img)
            mask_array.append(new_mask)
    print("Augmentation done")
    return img_array, mask_array


def load_slices(data, img_size, imgs, masks, n_images):
    print("\nStarting augmentation...")
    img_array = []
    mask_array = []
    for i in range(n_images):
        img = unpack(imgs[i])
        mask = unpack(masks[i])
        start_idx = int(img.shape[2] * 0.2)
        stop_idx = int(img.shape[2] * 0.8)
        new_dim = (img_size, img_size)
        for j in range(start_idx, stop_idx):
            new_img = cv2.resize(img[..., j], new_dim).astype('uint8')
            new_mask = cv2.resize(mask[..., j], new_dim).astype('uint8')
            img_array.append(new_img)
            mask_array.append(new_mask)
    print("Augmentation done")
    return img_array, mask_array


def aug_slices(data, img_size, imgs, masks, n_images):
    print("\nStarting augmentation...")
    img_array = []
    mask_array = []
    for i in range(n_images):
        img = unpack(imgs[i])
        mask = unpack(masks[i])
        dim = img.shape[2]
        new_dim = (img_size, img_size)
        for j in range(dim):
            new_img = cv2.resize(img[..., j], new_dim).astype('uint8')
            new_mask = cv2.resize(mask[..., j], new_dim).astype('uint8')
            img_array.append(new_img)
            mask_array.append(new_mask)
    print("Augmentation done")
    return img_array, mask_array
