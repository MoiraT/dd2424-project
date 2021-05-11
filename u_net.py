from tensorflow import keras
from keras.layers import *
from keras.models import *

def Unet(input_size = (299, 299, 1)):
    inputs = Input(input_size)
    
    #first level contracting path  
    conv1_1 = Conv2D(filter=64, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(input)
    conv1_2 = Conv2D(filter=64, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(conv1_1)
    pool1_1 = MaxPooling2D(pool_size=(2,2))(conv1_2)
    pool1_2 = Dropout(0.1)(pool1_1)

    #second level contracting path 
    conv2_1 = Conv2D(filter=64*2, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(pool1_2)
    conv2_2 = Conv2D(filter=64*2, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(conv2_1)
    pool2_1 = MaxPooling2D(pool_size=(2,2))(conv2_2)
    pool2_2 = Dropout(0.1)(pool2_1)

    #third level contracting path 
    conv3_1 = Conv2D(filter=64*4, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(pool2_2)
    conv3_2 = Conv2D(filter=64*4, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(conv3_1)
    pool3_1 = MaxPooling2D(pool_size=(2,2))(conv3_2)
    pool3_2 = Dropout(0.1)(pool3_1)

    #fouth level contracting path 
    conv4_1 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(pool3_2)
    conv4_2 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2,2))(conv4_2)
    pool4_2 = Dropout(0.1)(pool4_1)

    #fifth level bottom layer
    conv5_1 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(pool4_2)
    conv5_2 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(conv5_1)

    #fourth level expansive path
    #conv2dtranspose instead of upsamling, do the same but conv2dtranspose is smarter but requires more resources
    deconv4_1 = Conv2DTranspose(filter=64*8,  kernel_size=(3,3), strides=(2,2), activation="relu", padding=same, kernel_initializer="he_normal")(pool5_2)
    econv4_2 = concatenate([deconv4_1, conv4_2])
    epool4_2 = Dropout(0.1)(econv4_2)
    econv4_1 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(epool4_2)
    econv4_2 = Conv2D(filter=64*8, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(econv4_1)

    #third level espansive path
    deconv3_1 = Conv2DTranspose(filter=64*4,  kernel_size=(3,3), strides=(2,2), activation="relu", padding=same, kernel_initializer="he_normal")(econv4_2)
    econv3_2 = concatenate([deconv3_1, conv3_2])
    epool3_2 = Dropout(0.1)(econv3_2)
    econv3_1 = Conv2D(filter=64*4, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(epool3_2)
    econv3_2 = Conv2D(filter=64*4, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(econv3_1)

    #second level espansive path
    deconv2_1 = Conv2DTranspose(filter=64*2,  kernel_size=(3,3), strides=(2,2), activation="relu", padding=same, kernel_initializer="he_normal")(econv3_2)
    econv2_2 = concatenate([deconv2_1, conv2_2])
    epool2_2 = Dropout(0.1)(econv2_2)
    econv2_1 = Conv2D(filter=64*2, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(epool2_2)
    econv2_2 = Conv2D(filter=64*2, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(econv2_1)

    #first level espansive path
    deconv1_1 = Conv2DTranspose(filter=64*1,  kernel_size=(3,3), strides=(2,2), activation="relu", padding=same, kernel_initializer="he_normal")(econv2_2)
    econv1_2 = concatenate([deconv1_1, conv1_2])
    epool1_2 = Dropout(0.1)(econv1_2)
    econv1_1 = Conv2D(filter=64*1, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(epool1_2)
    econv1_2 = Conv2D(filter=64*1, kernel_size=(3,3), activation="relu", padding=same, kernel_initializer="he_normal")(econv1_1)
    #last step
    outputs = Conv2D(filter=1, kernel_size=(1,1), activation="sigmoid", padding=same)(econv1_2)

    model = Model(input = inputs, output = outputs)













