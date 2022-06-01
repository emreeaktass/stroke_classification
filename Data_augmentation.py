import albumentations as A
import numpy as np
from glob import glob
import cv2

hor_flip = A.Compose([
    A.HorizontalFlip(p=0.5),
])

rotate = A.Compose([
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=0.5),
])

shift = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.5),
])

elastic = A.Compose([
    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, p=0.5),
])

gauss_noise = A.Compose([
    A.GaussNoise(var_limit=(10, 100), p=0.5),
])

brightness = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
])

all_in_one = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(border_mode=cv2.BORDER_CONSTANT, limit=(-15, 15), p=0.5),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=(-5, 5), border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.GaussNoise(var_limit=(10, 100), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

if __name__ == '__main__':
    path_data = '../work/spl/train/data/'
    path_mask1 = '../work/spl/train/label1/'
    path_mask2 = '../work/spl/train/label2/'
    path_mask3 = '../work/spl/train/label3/'
    path_mask4 = '../work/spl/train/label4/'
    file_names_data = glob(path_data + '*')
    file_names_mask1 = glob(path_mask1 + '*')
    file_names_mask2 = glob(path_mask2 + '*')
    file_names_mask3 = glob(path_mask3 + '*')
    file_names_mask4 = glob(path_mask4 + '*')
    file_names_data.sort()
    file_names_mask1.sort()
    file_names_mask2.sort()
    file_names_mask3.sort()
    file_names_mask4.sort()
    for i, j, k, l, m in zip(file_names_data, file_names_mask1, file_names_mask2, file_names_mask3, file_names_mask4):
        print('..' + i.split('.')[2], '..' + j.split('.')[2])
        data = np.load(i)
        lab1 = np.load(j)
        lab2 = np.load(k)
        lab3 = np.load(l)
        lab4 = np.load(m)

        transformed = all_in_one(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_all_in_one', transformed_image)
        np.save('..' + j.split('.')[2] + '_all_in_one', lab1)
        np.save('..' + k.split('.')[2] + '_all_in_one', lab2)
        np.save('..' + l.split('.')[2] + '_all_in_one', lab3)
        np.save('..' + m.split('.')[2] + '_all_in_one', lab4)

        transformed = brightness(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_brightness', transformed_image)
        np.save('..' + j.split('.')[2] + '_brightness', lab1)
        np.save('..' + k.split('.')[2] + '_brightness', lab2)
        np.save('..' + l.split('.')[2] + '_brightness', lab3)
        np.save('..' + m.split('.')[2] + '_brightness', lab4)

        transformed = gauss_noise(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_gauss_noise', transformed_image)
        np.save('..' + j.split('.')[2] + '_gauss_noise', lab1)
        np.save('..' + k.split('.')[2] + '_gauss_noise', lab2)
        np.save('..' + l.split('.')[2] + '_gauss_noise', lab3)
        np.save('..' + m.split('.')[2] + '_gauss_noise', lab4)

        transformed = elastic(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_elastic', transformed_image)
        np.save('..' + j.split('.')[2] + '_elastic', lab1)
        np.save('..' + k.split('.')[2] + '_elastic', lab2)
        np.save('..' + l.split('.')[2] + '_elastic', lab3)
        np.save('..' + m.split('.')[2] + '_elastic', lab4)

        transformed = shift(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_shift', transformed_image)
        np.save('..' + j.split('.')[2] + '_shift', lab1)
        np.save('..' + k.split('.')[2] + '_shift', lab2)
        np.save('..' + l.split('.')[2] + '_shift', lab3)
        np.save('..' + m.split('.')[2] + '_shift', lab4)

        transformed = rotate(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_rotate', transformed_image)
        np.save('..' + j.split('.')[2] + '_rotate', lab1)
        np.save('..' + k.split('.')[2] + '_rotate', lab2)
        np.save('..' + l.split('.')[2] + '_rotate', lab3)
        np.save('..' + m.split('.')[2] + '_rotate', lab4)

        transformed = hor_flip(image=data)
        transformed_image = transformed["image"]

        np.save('..' + i.split('.')[2] + '_hor_flip', transformed_image)
        np.save('..' + j.split('.')[2] + '_hor_flip', lab1)
        np.save('..' + k.split('.')[2] + '_hor_flip', lab2)
        np.save('..' + l.split('.')[2] + '_hor_flip', lab3)
        np.save('..' + m.split('.')[2] + '_hor_flip', lab4)
