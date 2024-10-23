import torch
from torchvision import models
import torchvision.transforms as transforms

import cv2
from PIL import Image

import numpy as np

TRANSFORMS_PIPELINE = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(path,resize_shape=None) -> np.ndarray:
    img = np.array(Image.open(path).convert("RGB"))[...,:3]
    if resize_shape == None:
        return np.float32(img)/255
    return np.float32(cv2.resize(img,resize_shape))/255

def load_image_into_resnet_tensor(path,image_transform_pipeline=TRANSFORMS_PIPELINE,device="cpu"):
    return image_transform_pipeline(
        Image.open(path).convert("RGB")
    ).unsqueeze(0).to(device)


#https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True).eval()
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]
    
def get_model():
    return ResnetFeatureExtractor().to(DEVICE)

def calculate_cosine(path1,path2,model):
    img1 = load_image_into_resnet_tensor(path1,device=DEVICE)
    img2 = load_image_into_resnet_tensor(path2,device=DEVICE)
    features1 = model(img1)
    features2 = model(img2)
    sim = torch.nn.functional.cosine_similarity(features1,features2,dim=1).cpu().item()
    del img1,img2,features1,features2
    return sim



