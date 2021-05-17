from skimage import io


def crop_images(N):
    img_dim = 299  # The lung scan dataset image dimensions
    new_dim = 256

    for i in range(1, N+1):
        path = "./dataset/COVID-19_Radiography_Database/COVID-19_Radiography_Dataset/COVID/" + \
            "COVID-" + str(i) + ".png"
        img = io.imread(path)
        img = img[img_dim - new_dim: img_dim, img_dim - new_dim: img_dim]
        if img.shape != (256, 256):
            raise ValueError("Image dimensions are wrong")
        io.imsave("./dataset/processed_images/" + str(i) + ".tif", img)
    print("\nImages cropped and converted")
