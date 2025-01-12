from common import ComparisonModule

import numpy as np

import cv2

from scipy.stats import wasserstein_distance


class Color(ComparisonModule):
    def __init__(self):
        super().__init__("Color")

    def get_hue_histogram(self, image, num_bins):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image_hsv], [0], None, [num_bins], [0, 180])

        hist = hist / np.sum(hist)

        return hist

    def get_saturation_histogram(self, image, num_bins):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image_hsv], [1], None, [num_bins], [0, 256])

        hist = hist / np.sum(hist)

        return hist

    def get_hue_saturation_histogram(self, image, num_bins):
        return (
            self.get_hue_histogram(image, num_bins),
            self.get_saturation_histogram(image, num_bins),
        )

    def get_wasserstein_distance(self, hist1, hist2, num_bins):
        hist1_flat = hist1.flatten()
        hist2_flat = hist2.flatten()

        dist = wasserstein_distance(
            u_values=range(len(hist1_flat)),  # positions in the histogram
            v_values=range(len(hist2_flat)),
            u_weights=hist1_flat,  # weights for each position
            v_weights=hist2_flat,
        )

        return dist / num_bins

    def calculate_similarity(self, x: np.ndarray, y: np.ndarray) -> float:

        x_hue_hist, x_sat_hist = self.get_hue_saturation_histogram(x, 64)
        y_hue_hist, y_sat_hist = self.get_hue_saturation_histogram(y, 64)

        hue_dist = self.get_wasserstein_distance(x_hue_hist, y_hue_hist, 64)
        sat_dist = self.get_wasserstein_distance(x_sat_hist, y_sat_hist, 64)

        return 1 - (0.66 * hue_dist + 0.33 * sat_dist)
