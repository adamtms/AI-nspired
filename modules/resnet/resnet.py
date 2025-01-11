from common import ComparisonModule

import numpy as np

import torch
from torchvision import models
import torchvision.transforms as transforms


# https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True).eval()
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]


class Resnet(ComparisonModule):
    def __init__(self, use_cuda=True):
        super().__init__("ResNet")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )
        self.model = ResnetFeatureExtractor().to(self.device)
        self.transforms_pipeline = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def calculate_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        features_1, features_2 = self.model(x), self.model(y)
        sim = torch.nn.functional.cosine_similarity(features_1, features_2)
        del features_1, features_2
        return (sim.item() + 1) / 2


__all__ = ["Resnet"]
