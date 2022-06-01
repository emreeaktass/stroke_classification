import numpy as np
from glob import glob


if __name__ == '__main__':
    path_data = '../work/cls/KANAMA/data/'
    path_mask = '../work/cls/KANAMA/label/'
    file_names_data = glob(path_data + '*')
    file_names_mask = glob(path_mask + '*')
    file_names_data.sort()
    file_names_mask.sort()
    for i,j in zip(file_names_data, file_names_mask):
        print('..' + i.split('.')[2], '..' + j.split('.')[2])
        data = np.load(i)
        lab = np.array([0, 0, 1], dtype=np.uint8)

        np.save('..' + j.split('.')[2], lab)


# def smooth_labels(labels, factor=0.1):
#     # smooth the labels
#     labels *= (1 - factor)
#     labels += (factor / labels.shape[0])
#     # returned the smoothed labels
#     return labels
#
#
#
# lab = np.array([1., 0., 0., 0., 0., 0.])
#
# print(smooth_labels(lab))
# print(np.array(lab) * 0.9)