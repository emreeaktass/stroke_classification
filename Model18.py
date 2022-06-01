from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, UpSampling2D, concatenate, Flatten, Dense, \
    BatchNormalization, Lambda, GlobalAveragePooling2D, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
import tensorflow as tf
import sys


def unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    a = tf.keras.layers.Resizing(64, 64)(inputs)

    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(a)
    # conv1 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.2)(pool1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop1)
    # conv2 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.2)(pool2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop2)
    # conv3 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.2)(pool3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop3)
    # conv4 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.2)(pool4)

    # conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    # # conv4 = Conv2D(1, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # drop5 = Dropout(0.2)(pool5)

    #
    # conv9 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv9)
    # drop5 = Dropout(0.2)(pool5)

    gap = GlobalAveragePooling2D()(drop4)
    # flat = Flatten()(drop2)

    # dns1 = Dense(256, activation='relu', kernel_initializer='he_normal')(gap)

    dns2 = Dense(16, activation='relu', kernel_initializer='he_normal')(gap)

    dns3 = Dense(8, activation='relu', kernel_initializer='he_normal')(dns2)
    dns4 = Dense(4, activation='relu', kernel_initializer='he_normal')(dns3)

    otpt = Dense(1, activation='sigmoid')(dns4)

    model = Model(inputs, otpt)
    print(model.summary())
    # model.compile(optimizer=Adam(learning_rate=0.009), loss=tf.keras.losses.KLDivergence(), metrics=['accuracy'])
    # model.compile(optimizer=RMSprop(learning_rate=0.0011),
    #               loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    return model


def my_coef(y_true, y_pred):
    # y_true = tf.reshape(y_true, [32, 3])
    # y_pred = tf.reshape(y_pred, [32, 3])
    dice = 0
    s = 0.0
    dice = (tf.linalg.matmul(y_true, tf.transpose(y_pred)))
    for i in range(len(y_true)):
        for j in range(3):
            if y_true[i][j] == 0:
                s = s + y_pred[i][j] * -1
            else:
                s = s + y_pred[i][j] * 1
    # dice = tf.reduce_sum(dice)
    # print(tf.shape(y_true))
    # tf.print(s, output_stream=sys.stderr)
    s = s / 32.
    return s


def my_loss(y_true, y_pred):
    return 1 - my_coef(y_true, y_pred)
