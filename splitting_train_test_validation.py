import random
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
import time


def smooth_labels(labels, factor=0.1):
    lab = labels.copy()
    # print(lab)
    lab *= (1 - factor)
    lab += (factor / labels.shape[0])
    # print(lab)
    return np.array(lab)

if __name__ == '__main__':
    path_data = '../work/dataset/KANAMA/data/'
    path_mask = '../work/dataset/KANAMA/label/'
    file_names_data = glob(path_data + '*')
    file_names_mask = glob(path_mask + '*')
    file_names_data.sort()
    file_names_mask.sort()
    # print(len(file_names_data))
    datas = file_names_data
    labels = file_names_mask
    X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.1, random_state=47)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=47)
    # [1, 0] inme yok
    # [0, 1] inme var


    t_path = '../work/spl/test/'
    v_path = '../work/spl/validation/'
    tr_path = '../work/spl/train/'
    lab1 = np.array(0, dtype=np.uint8)
    lab2 = np.array(1, dtype=np.uint8)
    lab3 = np.array(0, dtype=np.uint8)
    lab4 = np.array(1, dtype=np.uint8)

    for i,j in zip(X_test, y_test):
        np.save(t_path + 'data/' + i.split('/')[-1].split('.')[0], np.load(i))
        np.save(t_path + 'label1/' + i.split('/')[-1].split('.')[0], lab1)
        np.save(t_path + 'label2/' + i.split('/')[-1].split('.')[0], lab2)
        np.save(t_path + 'label3/' + i.split('/')[-1].split('.')[0], lab3)
        np.save(t_path + 'label4/' + i.split('/')[-1].split('.')[0], lab4)

    for i,j in zip(X_val, y_val):
        np.save(v_path + 'data/' + i.split('/')[-1].split('.')[0], np.load(i))
        np.save(v_path + 'label1/' + i.split('/')[-1].split('.')[0], lab1)
        np.save(v_path + 'label2/' + i.split('/')[-1].split('.')[0], lab2)
        np.save(v_path + 'label3/' + i.split('/')[-1].split('.')[0], lab3)
        np.save(v_path + 'label4/' + i.split('/')[-1].split('.')[0], lab4)

    for i,j in zip(X_train, y_train):
        np.save(tr_path + 'data/' + i.split('/')[-1].split('.')[0], np.load(i))
        np.save(tr_path + 'label1/' + i.split('/')[-1].split('.')[0], lab1)
        np.save(tr_path + 'label2/' + i.split('/')[-1].split('.')[0], lab2)
        np.save(tr_path + 'label3/' + i.split('/')[-1].split('.')[0], lab3)
        np.save(tr_path + 'label4/' + i.split('/')[-1].split('.')[0], lab4)

# label 1 sağlıklı ve diğerleri
# label 2 sağlıksız ve diğerleri
# label 3 iskemi ve diğerleri
# label 4 kanama ve diğerleri