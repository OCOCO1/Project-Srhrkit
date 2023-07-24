from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D

model = Sequential()
# Layer 1: Depthwise Separable Convolution
model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())

# Layer 2: Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Layer 3: Depthwise Separable Convolution
model.add(DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same'))
model.add(BatchNormalization())
model.add(ReLU())

# Layer 4: Pooling Layer
model.add(MaxPool2D(pool_size=(2, 2)))

# Layer 5: Global Average Pooling
model.add(GlobalAveragePooling2D())

# Layer 6: Dense Layer (Hidden Layer)
model.add(Dense(256, activation='relu'))

# Layer 7: Dense Layer (Output Layer)
model.add(Dense(10, activation='softmax'))
