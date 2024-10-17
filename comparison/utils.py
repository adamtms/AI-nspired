import torch
from torchvision import models
import numpy as np

from matplotlib import pyplot as plt
import cv2
from PIL import Image

import torchvision.transforms as transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

INPUT_SHAPE = (256,256)

def load_img(path,resize_shape=None) -> np.ndarray:
    img = cv2.imread(path)
    if resize_shape == None:
        return np.float32(img)/255
    return np.float32(cv2.resize(img,resize_shape))/255

def load_image_into_resnet_tensor(path,device="cpu"):
    image_transform_pipeline = transforms.Compose([
        transforms.ToTensor()
    ])
    return image_transform_pipeline(
        Image.open(path).convert("RGB")
    ).unsqueeze(0).to(device)


#https://jacobgil.github.io/pytorch-gradcam-book/Pixel%20Attribution%20for%20embeddings.html
class ResnetFeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super(ResnetFeatureExtractor, self).__init__()
        self.model = model
        self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
                
    def __call__(self, x):
        return self.feature_extractor(x)[:, :, 0, 0]

class SimilarityTarget(RawScoresOutputTarget):
    def __init__(self,features) -> None:
        super().__init__()
        self.features = features
    def __call__(self, model_output):
        return torch.nn.functional.cosine_similarity(model_output,self.features,dim=0)
    
    
def make_comparison_plot_similarity(inspiration_path, final_image_path,model:ResnetFeatureExtractor,device):
    inspiration_tensor = load_image_into_resnet_tensor(inspiration_path,device)
    final_image_tensor = load_image_into_resnet_tensor(final_image_path,device)
    
    features_inspiration = model(inspiration_tensor)[0,:]
    features_final = model(final_image_tensor)[0,:]

    cosine_similarity = np.float32(torch.nn.functional.cosine_similarity(features_inspiration,features_final,dim=0).cpu().detach().numpy())

    target_layers = [
        #*model.model.layer1,
        #*model.model.layer2,
        #*model.model.layer3,
        model.model.layer4[-1]
        ]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam_final = cam(input_tensor=final_image_tensor, targets=[SimilarityTarget(features_inspiration)])
    grayscale_cam_inspiration = cam(input_tensor=inspiration_tensor, targets=[SimilarityTarget(features_final)])

    inspiration = load_img(inspiration_path)
    final_image = load_img(final_image_path)

    ax = plt.figure(figsize=(20,20))
    plt.subplot(3,2,1)
    plt.imshow(inspiration)
    plt.title("Inspiration")

    plt.subplot(3,2,2)
    plt.imshow(final_image)
    plt.title("Final Image")

    plt.subplot(3,2,3)
    plt.imshow(
        show_cam_on_image(inspiration, cv2.resize(grayscale_cam_inspiration[0],inspiration.shape[:2][::-1]), use_rgb=True)
        )
    plt.title("Inspiration Heatmap\n(What makes inspiration similar to final)")

    plt.subplot(3,2,4)
    plt.imshow(
        show_cam_on_image(final_image, cv2.resize(grayscale_cam_final[0],final_image.shape[:2][::-1]), use_rgb=True)
        )
    plt.title("Final Image Heatmap\n(What makes final similar to inspiration)")
    
    plt.suptitle(f"Similarity (cos): {cosine_similarity}")

    torch.cuda.empty_cache()