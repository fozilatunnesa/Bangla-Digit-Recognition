from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=20,
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = False)
val_datagen = ImageDataGenerator(  rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip = False)
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory('D://Compressed//BanglaLekha-Isolated//BanglaDigit//test',
                                            target_size=(28, 28),
                                            color_mode='grayscale',
                                            classes=['0','1', '2', '3', '4', '5','6', '7', '8', '9'],
                                            batch_size=32,
                                            interpolation="lanczos",
                                            class_mode='categorical',
                                            shuffle=False
                                            )

train_set = train_datagen.flow_from_directory('D://Compressed//BanglaLekha-Isolated//BanglaDigit//train',
                                              target_size=(28, 28),
                                              color_mode='grayscale',
                                              classes=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                              batch_size=32,
                                              interpolation="lanczos",
                                              class_mode='categorical')

validation_set = val_datagen.flow_from_directory('D://Compressed//BanglaLekha-Isolated//BanglaDigit//validation',
                                                 target_size=(28, 28),
                                                 color_mode='grayscale',
                                                 classes=['0','1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                 batch_size=32,
                                                 interpolation="lanczos",
                                                 class_mode='categorical')
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[28,28, 1]))
model.add(MaxPooling2D(pool_size=2, strides=2))

model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2, strides=2))
# fully-connected layer
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
# softmax classifier
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history= model.fit(train_set,
                    epochs=34,
                    verbose=2,
                    validation_data=validation_set,
                    steps_per_epoch=train_set.samples // 32,
                    validation_steps = validation_set.samples // 32)

print("model trained Successfully!")
_, accuracy = model.evaluate(train_set, verbose=1)
print('Accuracy of train set: %.2f' % (accuracy * 100))
_, accuracy = model.evaluate(validation_set, verbose=1)
print('Accuracy of validation set: %.2f' % (accuracy * 100))
_, accuracy = model.evaluate(test_set, verbose=1)
print('Accuracy of test set: %.2f' % (accuracy * 100))

# Save the model weights for future reference
model.save('cnn_model.h5')
