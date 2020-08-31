import os
import json
import glob
import argparse
import random
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from scipy.ndimage.filters import maximum_filter
# from skimage.feature import peak_local_max
from PIL import Image
import matplotlib.pyplot as plt

def load_data():

    images_url = r"C:\Users\RENT\miri\projects\mobileye\מובילאי\leftImg8bit_trainvaltest\leftImg8bit\train\aachen"
    image_list = glob.glob(os.path.join(images_url, '*_leftImg8bit.png'))

    labels_url = r"C:\Users\RENT\miri\projects\mobileye\מובילאי\gtFine_trainvaltest\gtFine\train\aachen"
    label_list = glob.glob(os.path.join(labels_url, "*_labelIds.png"))

    return image_list, label_list


def pass_over_lists(image_list, label_list):

    for i in range(len(image_list)):
        crop_tfl(image_list[i], label_list[i]) 



def crop_image(image, x_coords, y_coords, size):

    x_coords += size
    y_coords += size

    index = random.randrange(len(x_coords))

    result_crop = image[x_coords[index] - size:x_coords[index] + size + 1, y_coords[index] - size:y_coords[index] + size + 1]

    return result_crop
    # plt.imshow(result_crop)
    # plt.show(block=True)

def crop_tfl(image_url, label_url): 

    image = np.array(Image.open(image_url))
    label = np.array(Image.open(label_url))

    pixels_of_tfl = np.where(label == 19)
    pixels_not_of_tfl = np.where(label != 19)

    height, width = label.shape
    padding_size = 40

    
    if(len(pixels_of_tfl[0])):
        

        v_padding = np.zeros((padding_size, width), int)
        label = np.vstack([v_padding, label, v_padding])

        h_padding = np.zeros((height + padding_size*2, padding_size), int)
        label = np.hstack([h_padding, label, h_padding])


        v_padding = np.zeros((padding_size, width, 3), int)
        image = np.vstack([v_padding, image, v_padding])

        h_padding = np.zeros((height + padding_size*2, padding_size, 3), int)
        image = np.hstack([h_padding, image, h_padding])

        
        data_file = r'code\\script\\dataset\\data.bin'
        labels_file = r'code\\script\\dataset\\labels.bin'
        is_tfl = 1
        is_not_tfl = 0

        with open(data_file, mode='wb') as data_obj:
            with open(labels_file, mode='wb') as labels_obj:
                result = crop_image(image, pixels_of_tfl[0], pixels_of_tfl[1], padding_size)
                result.tofile(data_obj)
                labels_obj.write(str(1))


                result = crop_image(image, pixels_not_of_tfl[0], pixels_not_of_tfl[1], padding_size)
                result.tofile(data_obj)
                labels_obj.write(str(0))




# print(img.shape)
# plt.imshow(img)
# plt.show(block=True)

def main():
    image_list, label_list =  load_data()
    pass_over_lists(image_list, label_list)
    file = r'code\\script\\dataset\\labels.bin'
    f = open(file, "rb")
    try:
        byte = f.read(1)
        print(byte)
        while byte != "":
            # Do stuff with byte.
            byte = f.read(1)
            print(byte)

    finally:
        f.close()

main()