from tensorflow import keras
from keras.layers import *
from keras.models import *


class ContrLayer:

    def __init__(self, input_layer, factor, activ_func, kernel_initializer, dropout_rate):
        self.conv1 = Conv2D(filter=64*factor, kernel_size=(3, 3), activation=activ_func,
                            padding=same, kernel_initializer=kernel_initializer)(input_layer)
        self.conv2 = Conv2D(filter=64*factor, kernel_size=(3, 3), activation=activ_func,
                            padding=same, kernel_initializer=kernel_initializer)(self.conv_1)
        self.pool = MaxPooling2D(pool_size=(2, 2))(self.conv2)
        self.output = Dropout(dropout_rate)(self.pool)


class ExpLayer:

    def __init__(self, input_layer, concat_layer, factor, activ_func, kernel_initializer, dropout_rate):

        self.deconv1 = Conv2DTranspose(filter=64*factor, kernel_size=(3, 3), strides=(
            2, 2), activation=activ_func, padding=same, kernel_initializer=kernel_initializer)(input_layer)
        self.concat = concatenate([concat_layer, self.deconv1])
        self.drop = Dropout(dropout_rate)(self.concat)
        self.conv1 = Conv2D(filter=64*factor, kernel_size=(3, 3), activation=activ_func,
                            padding=same, kernel_initializer=kernel_initializer)(self.drop)
        self.output = Conv2D(filter=64*factor, kernel_size=(3, 3), activation=activ_func,
                             padding=same, kernel_initializer=kernel_initializer)(self.conv1)


def unet(input_size):

    input_layer = Input(input_size)

    # contracting path
    layer1 = ContrLayer(input_layer, 1, "relu", "he_normal", 0.5)
    layer2 = ContrLayer(layer1.output, 2, "relu", "he_normal", 0.5)
    layer3 = ContrLayer(layer2.output, 4, "relu", "he_normal", 0.5)
    layer4 = ContrLayer(layer3.output, 8, "relu", "he_normal", 0.5)

    # bottom layer
    layer5_1 = Conv2D(filter=64*8, kernel_size=(3, 3), activation="relu",
                      padding=same, kernel_initializer="he_normal")(layer4.output)
    layer5_2 = Conv2D(filter=64*8, kernel_size=(3, 3), activation="relu",
                      padding=same, kernel_initializer="he_normal")(layer5_1)

    # expansive path
    layer4 = ExpLayer(layer5_2, layer4.conv2, 8, "relu", "he_normal", 0.5)
    layer3 = ExpLayer(layer4.output, layer3.conv2, 4, "relu", "he_normal", 0.5)
    layer2 = ExpLayer(layer3.output, layer2.conv2, 2, "relu", "he_normal", 0.5)
    layer1 = ExpLayer(layer2.output, layer1.conv2, 1, "relu", "he_normal", 0.5)

    output_layer = Conv2D(filter=1, kernel_size=(
        1, 1), activation="sigmoid", padding=same)(layer1.output)

    model = Model(input=input_layer, output=output_layer)

    return model
