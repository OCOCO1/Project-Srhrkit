from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Concatenate, Input
from tensorflow.keras.models import Model

def inception_module(x, filters):
    # 1x1 Convolution
    conv1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
    # 3x3 Convolution
    conv3x3 = Conv2D(filters[1], (3, 3), padding='same', activation='relu')(x)
    # 5x5 Convolution
    conv5x5 = Conv2D(filters[2], (5, 5), padding='same', activation='relu')(x)
    # Max Pooling
    maxpool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    maxpool = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate all the filters
    inception = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool])
    return inception

inputs = Input(shape=(32, 32, 3))

# Layer 1: Convolutional Layer
x = Conv2D(filters=32, kernel_size=(4, 4), activation='relu')(inputs)
# Layer 2: Pooling Layer
x = MaxPool2D(pool_size=(2, 2))(x)

# Layer 3: Inception Module
x = inception_module(x, filters=[32, 32, 32, 32])
# Layer 4: Pooling Layer
x = MaxPool2D(pool_size=(2, 2))(x)

# Layer 5: Inception Module
x = inception_module(x, filters=[64, 64, 64, 64])
# Layer 6: Flatten Layer
x = Flatten()(x)
# Layer 7: Dense Layer (Hidden Layer)
x = Dense(256, activation='relu')(x)
# Layer 8: Dense Layer (Output Layer)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
