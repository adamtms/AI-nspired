import torch
from torchvision import models
import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

import cv2
from PIL import Image

import numpy as np

from functools import lru_cache

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

class SimilarityTarget(RawScoresOutputTarget):
    def __init__(self,features) -> None:
        super().__init__()
        self.features = features
    def __call__(self, model_output):
        return torch.nn.functional.cosine_similarity(model_output,self.features,dim=0)

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

@lru_cache(maxsize=128)
def calculate_cosine(path1,path2,model):
    img1 = load_image_into_resnet_tensor(path1,device=DEVICE)
    img2 = load_image_into_resnet_tensor(path2,device=DEVICE)
    features1 = model(img1)
    features2 = model(img2)
    sim = torch.nn.functional.cosine_similarity(features1,features2,dim=1).cpu().item()
    del img1,img2,features1,features2
    return sim

def make_heatmap(path1:str,path2:str,save_path:str,model:ResnetFeatureExtractor):
    # Makes heatmap of what makes image1 similar to image2
    img1 = load_image_into_resnet_tensor(path1,device=DEVICE)
    img2 = load_image_into_resnet_tensor(path2,device=DEVICE)
    features2 = model(img2)[0,:]
    
    target_layers = [
        model.model.layer4[-1]
        ]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam= cam(input_tensor=img1, targets=[SimilarityTarget(features2)])
    
    img1_pixels = load_img(path1)
    cam_resized = cv2.resize(grayscale_cam[0],img1_pixels.shape[:2][::-1])
    heatmap_mask = np.dstack([cam_resized, cam_resized, cam_resized])
    
    heatmap_mask = (heatmap_mask - np.min(heatmap_mask))/(np.max(heatmap_mask)-np.min(heatmap_mask))
    
    img_masked = (img1_pixels*heatmap_mask*255).astype(np.uint8)
    
    cv2.imwrite(save_path,cv2.cvtColor(img_masked,cv2.COLOR_RGB2BGR))
    return save_path