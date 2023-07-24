from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
# Layer 1: Convolutional Layer
model.add(Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)))
# Layer 2: Max Pooling Layer
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
# Layer 3: Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
# Layer 4: Max Pooling Layer
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
# Layer 5: Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 6: Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 7: Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# Layer 8: Max Pooling Layer
model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))
# Layer 9: Flatten Layer
model.add(Flatten())
# Layer 10: Dense Layer (Hidden Layer)
model.add(Dense(4096, activation='relu'))
# Layer 11: Dense Layer (Hidden Layer)
model.add(Dense(4096, activation='relu'))
# Layer 12: Dense Layer (Output Layer)
model.add(Dense(1000, activation='softmax'))
