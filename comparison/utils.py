import torch
from torchvision import models
import numpy as np

from matplotlib import pyplot as plt
import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

def load_img(path,resize_shape=None) -> np.ndarray:
    img = cv2.imread(path)
    if resize_shape == None:
        return np.float32(img)/255
    return np.float32(cv2.resize(img,resize_shape))/255
    
def img_to_tensor(img,device):
    return preprocess_image(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ).to(device)

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
    
def make_comparison_plot(inspiration, final_image,model:ResnetFeatureExtractor,device):
    inspiration_tensor = img_to_tensor(inspiration,device)
    final_image_tensor = img_to_tensor(final_image,device)

    features_inspiration = model(inspiration_tensor)[0,:]
    features_final = model(final_image_tensor)[0,:]

    cosine_similarity = np.float32(torch.nn.functional.cosine_similarity(features_inspiration,features_final,dim=0).cpu().detach().numpy())

    target_layers = [model.model.layer4[-1]] # Last conv layer in ResNet
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam_final = cam(input_tensor=final_image_tensor, targets=[SimilarityTarget(features_inspiration)])
    grayscale_cam_inspiration = cam(input_tensor=inspiration_tensor, targets=[SimilarityTarget(features_final)])

    cam_image_inspiration = show_cam_on_image(inspiration, grayscale_cam_inspiration[0], use_rgb=True)
    cam_image_final = show_cam_on_image(final_image, grayscale_cam_final[0], use_rgb=True)

    print(cosine_similarity)
    ax = plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(inspiration)
    plt.title("Inspiration")

    plt.subplot(2,2,2)
    plt.imshow(final_image)
    plt.title("Final Image")

    plt.subplot(2,2,3)
    plt.imshow(cam_image_inspiration)
    plt.title("Inspiration")

    plt.subplot(2,2,4)
    plt.imshow(cam_image_final)
    plt.title("Final Image")

    torch.cuda.empty_cache()