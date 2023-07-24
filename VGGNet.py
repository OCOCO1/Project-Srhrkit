from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model = Sequential()
# Layer 1: Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
# Layer 2: Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
# Layer 3: Max Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 4: Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
# Layer 5: Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
# Layer 6: Max Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 7: Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
# Layer 8: Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
# Layer 9: Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
# Layer 10: Max Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 11: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 12: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 13: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 14: Max Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 15: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 16: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 17: Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), activation='relu'))
# Layer 18: Max Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 19: Flatten Layer
model.add(Flatten())
# Layer 20: Dense Layer (Hidden Layer)
model.add(Dense(4096, activation='relu'))
# Layer 21: Dense Layer (Hidden Layer)
model.add(Dense(4096, activation='relu'))
# Layer 22: Dense Layer (Output Layer)
model.add(Dense(1000, activation='softmax'))
