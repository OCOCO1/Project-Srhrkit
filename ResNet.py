from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Add, Input
from tensorflow.keras.models import Model

def conv_block(inputs, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

def identity_block(inputs, filters, kernel_size):
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(inputs)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = Add()([x, inputs])
    x = Activation('relu')(x)
    return x

inputs = Input(shape=(32, 32, 3))

# Layer 1: Convolutional Layer
x = Conv2D(filters=32, kernel_size=(4, 4), activation='relu')(inputs)
# Layer 2: Pooling Layer
x = MaxPool2D(pool_size=(2, 2))(x)

# Layer 3: Convolutional Block
x = conv_block(x, filters=32, kernel_size=(4, 4))
# Layer 4: Pooling Layer
x = MaxPool2D(pool_size=(2, 2))(x)

# Layer 5: Identity Block
x = identity_block(x, filters=32, kernel_size=(4, 4))
# Layer 6: Dense Layer (Hidden Layer)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
# Layer 7: Dense Layer (Output Layer)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
