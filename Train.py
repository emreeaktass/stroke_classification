import tensorflow as tf
import LRSchedular
from sklearn.model_selection import train_test_split
import Dataset as D
import DataLoader as DL
import Model18 as M
import os
import numpy as np
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# print(i[0])
# print(labels)

# returned the smoothed labels

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    X_train, y_train = D.get_splitted_datas('train')
    X_val, y_val = D.get_splitted_datas('validation')
    X_test, y_test = D.get_splitted_datas('test')
    train_generator = DL.get_generator(X_train, y_train, 16)
    val_generator = DL.get_generator(X_val, y_val, 16)
    test_generator = DL.get_generator(X_test, y_test, 1)


    # for i in val_generator:
    #     print(i[0].shape)
    # X_test = []
    # y_test = []
    # for i in test_generator:
    #     X_test.append(i[0][0])
    #     y_test.append(i[1][0])
    # X_test = np.array(X_test)
    # y_test = np.array(y_test)

    def scheduler(epoch, lr):
        if epoch % 2 == 0:
            return lr * 0.98
        else:
            return lr

    sch = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    clr = LRSchedular.CyclicLR(base_lr=1e-5, max_lr=1e-3,
                               step_size=4000., mode='triangular2')

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=30, verbose=1,
        mode='auto', baseline=None, restore_best_weights=True
    )

    # Xs = np.array(Xs)
    # ys = np.array(ys)
    # print(Xs.shape)
    # print(ys.shape)
    # model = M.unet()
    # model.fit(train_generator, validation_data=val_generator, epochs=5000, verbose=1, shuffle=False,
    #           callbacks=[sch, early_stop])
    # model.save('record24/')

    model = tf.keras.models.load_model('record21/', compile =False)
    #
    #
    X_test = []
    y_test = []
    for i in test_generator:
        X_test.append(i[0][0])
        y_test.append(i[1][0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # print(X_test.shape)
    y_pred = model.predict(X_test)
    counter = 0
    for i in range(len(y_pred)):
        # print(i)
        if np.round(y_pred[i]) == y_test[i]:

            counter = counter + 1

    print(counter / len(y_pred))


# OLD
# record1 validation 0.93 test 0.91
# record2 validation 0.92 test 0.89
# record7 test 0.9384
# record11 validation 0.9754 test 0.9353
# record12 validation 0.97 test 0.9384
# record13 validation 0.95 test 0.9476
# record14 validation0 0.9446 test 0.9323
# record15 validation .9477 test .9507 Model13
# record16 validation 0.97 test 0.9538 Model14 inmeyok inmevar 0-1
# record17 validation 0.96 test 0.9449 Model15 iskemi, kanama 0-1
# record18 validation 0.99 test 0.9541 Model16 iskemi, kanama 0-1
# record19 validation 0.97 test 0.9411 Model17 inmeyok, iskemi 0-1
# record20 validation 0.97 test 0.9702 Model18 inmeyok, kanama 0-1

# NEW
# record21 validation 0.96 test 0.9569 sağlıklı-diğerleri 1-0
# record22 validation 0.9569 test 0.9538 sağlıksız-diğerleri 1-0
# record23 validation 0.9877 test 0.96 iskemi-diğerleri 1-0
# record24 validation 0.9785 test 0.9846 kanama-diğerleri 1-0