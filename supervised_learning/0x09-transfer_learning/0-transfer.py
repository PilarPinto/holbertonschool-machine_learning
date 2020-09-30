#!/usr/bin/env python3
'''
CFAR 10 transfer
learning model
'''
import tensorflow.keras as K


classes = 10
epochs = 27
b_size = 128
verbo = 1


def preprocess_data(X, Y):
    '''Preprocessing data before
    used in the arch'''
    X_p = K.applications.vgg16.preprocess_input(X.astype('float32'))
    Y_p = K.utils.to_categorical(Y, classes)
    return X_p, Y_p


(X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
X_train, Y_train = preprocess_data(X_train, Y_train)
X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

vgg = K.applications.VGG16(include_top=False,
                           weights='imagenet', pooling='max')

output = vgg.layers[-1].output
output = K.layers.Flatten()(output)
vgg_model = K.Model(vgg.input, output)

vgg_model.trainable = True
set_trainable = False
for layer in vgg_model.layers:
    if layer.name in ['block5_conv1', 'block4_conv1']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = K.Sequential()
model.add(K.layers.UpSampling2D())
model.add(K.layers.BatchNormalization())
model.add(vgg_model)
model.add(K.layers.Dense(512, activation='relu'))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.3))
model.add(K.layers.Dense(classes, activation='softmax'))


opt = K.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt,
              loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=X_train, y=Y_train,
                    validation_data=(X_valid, Y_valid),
                    epochs=epochs,
                    batch_size=b_size,
                    verbose=verbo,
                    steps_per_epoch=100,
                    shuffle=True,
                    validation_steps=15)
model.summary()
model.save('cifar10.h5')
