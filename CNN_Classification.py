from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
(in_train, out_train), (in_test, out_test) = cifar10.load_data()

out_cat_train = to_categorical(out_train, 10)
out_cat_test = to_categorical(out_test, 10)

# Create Sequential Model
model = Sequential()
# Layer 1: Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu',))
# Layer 2: Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 3: Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32,32,3), activation='relu',))
# Layer 4: Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))
# Layer 5: Flatten Layer
model.add(Flatten())
# Layer 6: Dense Layer (Hidden Layer)
model.add(Dense(256, activation='relu'))
# Layer 7: Dense Layer (Output Layer)
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(in_train, out_cat_train, epochs=15, validation_data=(in_test, out_cat_test))

model.save('CNN_MODEL.h5')
