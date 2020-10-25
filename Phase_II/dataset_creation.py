import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Phase_I.run_attention import find_tfl_lights


def load_binary_file(data_dir: str, crop_shape=(81, 81)) -> dict:
    images = np.memmap(data_dir + '\\data.bin', mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(data_dir + '\\labels.bin', mode='r', dtype=np.uint8)

    return {'images': images, 'labels': labels}


def viz_my_data(images: np.memmap, labels: np.memmap, predictions=None, num=(5, 5),
                labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]

    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]
    idxs = np.random.randint(0, labels.shape[0], n)

    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None:
            title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)
    plt.show()


def load_tfl_data(url: str, suffix: str) -> list:
    data_list = []

    for subdir, dirs, files in os.walk(url):
        for directory in dirs:
            data_list += glob.glob(os.path.join(url + '\\' + directory, suffix))

    return data_list


def load_data(dir_set: str) -> dict:
    image_suffix = '*_leftImg8bit.png'
    label_suffix = '*_labelIds.png'

    url_image = "C:\\Users\\RENT\\miri\\projects\\mobileye\\מובילאי\\leftImg8bit_trainvaltest\\leftImg8bit\\"
    url_label = "C:\\Users\\RENT\\miri\\projects\\mobileye\\מובילאי\\gtFine_trainvaltest\\gtFine\\"

    return {'images': load_tfl_data(url_image + dir_set, image_suffix),
            'labels': load_tfl_data(url_label + dir_set, label_suffix)}


def open_image(image_url: str) -> np.array:
    return np.array(Image.open(image_url))


def padding_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    height, width, dim = image.shape

    v_padding = np.zeros((padding_size, width, dim), int)
    h_padding = np.zeros((height + padding_size * 2, padding_size, dim), int)

    image = np.vstack([v_padding, image, v_padding])
    image = np.hstack([h_padding, image, h_padding])

    return image


def get_rand_pixel(pixels: int):
    rand_p = random.choice(pixels)
    index_rand_p = pixels.index(rand_p)

    return rand_p, index_rand_p


def crop_image(image: np.ndarray, x, y, size: int):
    x += size
    y += size
    result_crop = image[x - size:x + size + 1, y - size:y + size + 1]

    return result_crop


def save_image(dir_name: str, image, label):
    data_root_path = "data/dataset"

    with open(f"{data_root_path}\\{dir_name}\\data.bin", "ab") as data_file:
        np.array(image, dtype=np.uint8).tofile(data_file)

    with open(f"{data_root_path}\\{dir_name}\\labels.bin", "ab") as labels_file:
        np.asarray([label], dtype=np.uint8).tofile(labels_file)


def call_write_to_bin_file(x_coords, y_coords, label, image, dir_name: str) -> None:
    pixels_of_tfl = [p for p in zip(y_coords, x_coords) if label[p[0], p[1]] == 19]
    pixels_not_of_tfl = [p for p in zip(y_coords, x_coords) if label[p[0], p[1]] != 19]
    size = 81
    count = 0

    image = padding_image(image, size // 2)

    while pixels_of_tfl and pixels_not_of_tfl and count < 3:
        count += 1

        rand_tfl, index_rand_tfl = get_rand_pixel(pixels_of_tfl)
        pixels_of_tfl = pixels_of_tfl[:index_rand_tfl] + pixels_of_tfl[index_rand_tfl + 1:]

        cropped_image = crop_image(image, rand_tfl[0], rand_tfl[1], size // 2)
        save_image(dir_name, cropped_image, 1)

        rand_not_tfl, index_rand_not_tfl = get_rand_pixel(pixels_not_of_tfl)
        pixels_not_of_tfl = pixels_not_of_tfl[:index_rand_not_tfl] + pixels_not_of_tfl[index_rand_not_tfl + 1:]

        cropped_image = crop_image(image, rand_not_tfl[0], rand_not_tfl[1], size // 2)
        save_image(dir_name, cropped_image, 0)


def create_dataset(dir_name: str, url_tuple: tuple) -> None:
    image = open_image(url_tuple[0])
    label = open_image(url_tuple[1])

    x_red, y_red, x_green, y_green = find_tfl_lights(image, some_threshold=42)
    plt.plot(x_red, y_red, 'ro', color='r', markersize=4)
    plt.plot(x_green, y_green, 'ro', color='g', markersize=4)

    x_list = np.concatenate([x_red, x_green])

    y_list = np.concatenate([y_red, y_green])
    call_write_to_bin_file(x_list.tolist(), y_list.tolist(), label, image, dir_name)


def main():
    dir_name_t = 'train'
    tfl_data_t = load_data(dir_name_t)

    for item in zip(*tfl_data_t.values()):
        create_dataset(dir_name_t, item)

    dir_name_v = 'val'
    tfl_data_v = load_data(dir_name_v)

    for item in zip(*tfl_data_v.values()):
        create_dataset(dir_name_v, item)

    train_path = "data/dataset/train"
    val_path = "data/dataset/val"

    dataset = {'train': load_binary_file(train_path), 'val': load_binary_file(val_path)}

    for k, v in dataset.items():
        print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

    viz_my_data(num=(6, 6), **dataset['train'])
    viz_my_data(num=(6, 6), **dataset['val'])


if __name__ == "__main__":
    main()
