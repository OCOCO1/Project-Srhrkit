# CNN
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
 
(in_train, out_train), (in_test, out_test) = cifar10.load_data()

type(in_train)

in_train.shape

in_train = in_train/255
in_test = in_test/255

type(out_train)

type(out_train)

out_train.shape

out_train[0]

out_cat_train = to_categorical(out_train, 10)
out_cat_test = to_categorical(out_test, 10)

out_train[0]
out_cat_train[0]

model = Sequential()

#layer 1 Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),
                 input_shape=(256,256,3),
                 activation="relu")
          )

#Layer 2 Pooling Layer 
model.add (MaxPool2D(pool_size=(5,5)))

#layer 3 Convolutional Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),
                 input_shape=(256,256,3),
                 activation="relu")
          )

#Layer 4 Pooling Layer 
model.add (MaxPool2D(pool_size=(5,5)))

#Layer 5 Flatten Layer
model.add(Flatten())

#Layer 6 Denes Layer (Hidden Layer)
model.add(Dense(256,activation="relu"))

#Layer 7 Dense Layer (Output Layer)
model.add(Dense(10, activation='softmax'))


#compile model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
model.fit(in_train, out_cat_train, epochs=15, validation_data=(in_test, out_cat_test))

early_stop = EarlyStopping(monitor='val_loss', patience=2)
model.fit(in_train, out_cat_train, epochs=15, validation_data=(in_test, out_cat_test), callbacks=[early_stop])
 #Model Evaluation
 metrics = pd.DataFrame(model.history.history)
 metrics
 prediction = model.predict_classes(in_test)
 print(classification_report(out_test, prediction))
 model.predict_classes(my_image.reshape(1,32,32,3))
