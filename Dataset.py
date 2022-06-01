import glob

from tensorflow.keras.utils import Sequence
import numpy as np
from sklearn.model_selection import train_test_split


class Dataset(Sequence):
    def __init__(self, data_root, label_root):
        self.data_root = data_root
        self.label_root = label_root

        self.data_file_names = glob.glob(data_root)
        self.label_file_names = glob.glob(label_root)

    def __getitem__(self, idx):

        return self.data_file_names[idx], self.label_file_names[idx]

    def __len__(self):
        return len(self.data_file_names)


def get_splitted_datas(name):
    data_root = '../work/spl/' + name + '/data/*'
    label_root = '../work/spl/' + name + '/label1/*'
    dataset = Dataset(data_root, label_root)
    datas = [i[0] for i in dataset]
    masks = [i[1] for i in dataset]
    return datas, masks


