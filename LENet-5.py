from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential()
# Layer 1: Convolutional Layer
model.add(Conv2D(filters=6, kernel_size=(5,5), activation='tanh', input_shape=(32,32,3)))
# Layer 2: Average Pooling Layer
model.add(AveragePooling2D(pool_size=(2,2)))
# Layer 3: Convolutional Layer
model.add(Conv2D(filters=16, kernel_size=(5,5), activation='tanh'))
# Layer 4: Average Pooling Layer
model.add(AveragePooling2D(pool_size=(2,2)))
# Layer 5: Flatten Layer
model.add(Flatten())
# Layer 6: Dense Layer (Hidden Layer)
model.add(Dense(120, activation='tanh'))
# Layer 7: Dense Layer (Hidden Layer)
model.add(Dense(84, activation='tanh'))
# Layer 8: Dense Layer (Output Layer)
model.add(Dense(10, activation='softmax'))
