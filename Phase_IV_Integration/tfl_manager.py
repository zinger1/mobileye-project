import pickle

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from Phase_I.run_attention import find_tfl_lights
from Phase_II.dataset_creation import crop_image, padding_image
from Phase_III.SFM import calc_TFL_dist
from Phase_III.SFM_standAlone import FrameContainer


class Frame:
    def __init__(self, frame_id, frame_path):
        self.img = np.array(Image.open(frame_path))
        self.img_path = frame_path
        self.frame_id = frame_id
        self.candidates = []
        self.auxiliary = []
        self.tfl_candidates = []
        self.tfl_auxiliary = []
        self.tfl_distance = []

    def initialized_data(self, x_list: list, y_list: list, color: str) -> None:
        for point in zip(x_list, y_list):
            self.candidates.append(point)
            self.auxiliary.append(color)

    def light_detection(self) -> None:
        x_red, y_red, x_green, y_green = find_tfl_lights(self.img, some_threshold=42)
        self.initialized_data(x_red, y_red, 'r')
        self.initialized_data(x_green, y_green, 'g')


def load_pickle_file(pickle_path: str):
    with open(pickle_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')

        return data


def visualize(prev_frame: Frame, curr_frame: Frame) -> None:
    fig, (candidate, traffic_light, dis) = plt.subplots(1, 3, figsize=(24, 12))
    candidate.set_title('candidates')
    candidate.imshow(Image.open(curr_frame.img_path))

    for idx, point in enumerate(prev_frame.candidates):
        candidate.plot(point[0], point[1], curr_frame.auxiliary[idx] + "+")

    traffic_light.set_title('traffic_lights')
    traffic_light.imshow(Image.open(curr_frame.img_path))

    for idx, point in enumerate(prev_frame.tfl_candidates):
        traffic_light.plot(point[0], point[1], curr_frame.tfl_auxiliary[idx] + "+")

    if curr_frame is not None:
        dis.set_title('distance')
        dis.imshow(Image.open(curr_frame.img_path))

        if len(curr_frame.tfl_distance):
            for idx, point in enumerate(curr_frame.tfl_candidates):
                dis.text(point[0], point[1], r'{0:.1f}'.format(curr_frame.tfl_distance[idx, 2]), color='b')
    plt.show(block=True)


class TFLManager:
    def __init__(self, pkl_path: str):
        self.net = tf.keras.models.load_model("../Phase_II/data/model.h5")
        self.pickle_data = load_pickle_file(pkl_path)
        self.pp = self.pickle_data['principle_point']
        self.focal = self.pickle_data['flx']

        self.prev_frame = None
        self.current_frame = None

        self.prev_container = None
        self.current_container = None

    def tfl_detection(self) -> None:
        size = 40
        img = padding_image(self.current_frame.img, size)

        for idx, point in enumerate(self.current_frame.candidates):
            x = point[0] + size
            y = point[1] + size
            cropped_img = crop_image(img, y, x, size)
            crop_shape = (81, 81)
            l_predictions = self.net.predict(cropped_img.reshape([-1] + list(crop_shape) + [3]))

            if l_predictions[0][0] > 0.97:
                self.current_frame.tfl_candidates.append(point)
                self.current_frame.tfl_auxiliary.append(self.current_frame.auxiliary[idx])

    def tfl_distance(self) -> np.array:
        self.prev_container = FrameContainer(self.prev_frame.img_path)
        self.prev_container.traffic_light = self.prev_frame.tfl_candidates
        self.current_container = FrameContainer(self.current_frame.img_path)
        self.current_container.traffic_light = self.current_frame.tfl_candidates
        self.load_EM_list(self.current_frame.frame_id)
        self.current_container = calc_TFL_dist(self.prev_container, self.current_container, self.focal, self.pp)

        return np.array(self.current_container.traffic_lights_3d_location)

    def load_EM_list(self, frame_id: int) -> None:
        self.current_container.EM = np.eye(4)
        self.current_container.EM = self.pickle_data['egomotion_' + str(frame_id - 1) + '-' + str(frame_id)]

    def on_frame(self, img_id: int, img_path: str) -> None:
        self.current_frame = Frame(img_id, img_path)
        self.current_frame.light_detection()
        self.tfl_detection()

        if self.prev_frame is not None:
            self.current_frame.tfl_distance = self.tfl_distance()
        self.prev_frame = self.current_frame

        visualize(self.prev_frame, self.current_frame)
