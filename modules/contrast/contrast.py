from common import ComparisonModule

import cv2

import numpy as np


class Contrast(ComparisonModule):
    def __init__(self):
        super().__init__("Contrast")

    def rmse(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.mean((x1 - x2) ** 2)) / np.sqrt(len(x1))

    def ame(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.mean(np.abs(x1 - x2))

    def get_entropy(self, image, num_bins=64):
        histogram = np.histogram(image, bins=num_bins, range=(0, 1))[0] / image.size

        # Remove zero probabilities (log2(0) is undefined)
        histogram = histogram[histogram > 0]

        # Calculate entropy
        entropy = -np.sum(histogram * np.log2(histogram))

        max_entropy = np.log2(num_bins)

        return entropy / max_entropy

    def get_contrast_features(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        image /= 255.0
        l = image[:, :, 0]
        features = [
            np.mean(l),
            # np.median(l),
            np.std(l),
            (np.max(l) - np.min(l)) / (np.max(l) + np.min(l)),
            self.get_entropy(l),
        ]
        return features

    def calculate_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        features1 = self.get_contrast_features(x)
        features2 = self.get_contrast_features(y)
        difference = self.rmse(features1, features2)
        return 1 - difference
