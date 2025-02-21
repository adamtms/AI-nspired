import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import ConnectionPatch
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dinov2.models.vision_transformer import vit_base

class DinoV2():
    
    def __init__(self, 
                checkpoint: str ='dinov2_vitb14_reg4_pretrain.pth', 
                patch_size: int = 14, 
                img_size: int = 526, 
                n_register_tokens: int = 4, 
                smaller_edge_size: int = 448, 
                device='cuda' if torch.cuda.is_available() else 'cpu'
                ):
        self.model = vit_base(
            patch_size=patch_size,
            img_size=img_size,
            init_values=1.0,
            num_register_tokens=n_register_tokens,
            block_chunks=0
        )
        self.patch_size = patch_size
        self.smaller_edge = smaller_edge_size
        self.n_register_tokens = n_register_tokens
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(size=self.smaller_edge, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])
        self.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(self.device)
        self.model.eval()

    def prepare_image(self, rgb_image_numpy):
        with torch.inference_mode():
            image = Image.fromarray(rgb_image_numpy)
            image_tensor = self.transform(Image.fromarray(rgb_image_numpy))
            resize_scale = image.width / image_tensor.shape[2]
            del rgb_image_numpy
            torch.cuda.empty_cache()

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:] # C x H x W
        cropped_width, cropped_height = width - width % self.patch_size, height - height % self.patch_size # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.patch_size, cropped_width // self.patch_size)
            
        return image_tensor, grid_size, resize_scale
    
    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask

    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        return row, col
  
    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens

    def extract_features(self, image_numpy, pooling: bool = True):
        with torch.inference_mode():
            image_tensor = self.prepare_image(image_numpy)[0]
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_tensor)[0].squeeze()
            del image_tensor, image_numpy
            torch.cuda.empty_cache()

            if pooling == False:
                return tokens.cpu().numpy()

            pooled_features = tokens.mean(dim=0)
            del tokens
            torch.cuda.empty_cache()

            return pooled_features

    def calculate_similarity(self, image1: str, image2: str):
        with torch.inference_mode():
            features1 = self.extract_features(cv2.cvtColor(cv2.imread(image1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            features2 = self.extract_features(cv2.cvtColor(cv2.imread(image2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

            similarity = F.cosine_similarity(features1, features2, dim=0)
            del features1, features2
            torch.cuda.empty_cache()

            return (similarity.item() + 1) / 2

    def create_attention_mask(self, image_metric, save: bool = False, show: bool = False):
        with torch.inference_mode():
            normalized_metric = Normalize(vmin=image_metric.min(), vmax=image_metric.max())(image_metric)
            del image_metric
            torch.cuda.empty_cache()

            # Apply the Reds colormap
            reds = plt.cm.Reds(normalized_metric)

            # Create the alpha channel
            alpha_max_value = 1.00  # Set your max alpha value

            # Adjust this value as needed to enhance lower values visibility
            gamma = 0.5  

            # Apply gamma transformation to enhance lower values
            enhanced_metric = np.power(normalized_metric, gamma)
            del normalized_metric, gamma
            torch.cuda.empty_cache()

            # Create the alpha channel with enhanced visibility for lower values
            alpha_channel = enhanced_metric * alpha_max_value

            # Add the alpha channel to the RGB data
            rgba_mask = np.zeros((enhanced_metric.shape[0], enhanced_metric.shape[1], 4))
            rgba_mask[..., :3] = reds[..., :3]  # RGB
            rgba_mask[..., 3] = alpha_channel  # Alpha
            del reds, alpha_max_value, enhanced_metric, alpha_channel
            torch.cuda.empty_cache()
            
            # Convert the numpy array to PIL Image
            rgba_image = Image.fromarray((rgba_mask * 255).astype(np.uint8))
            del rgba_mask
            torch.cuda.empty_cache()

            if save:
                rgba_image.save('attention_mask.png')
            if show:
                plt.imshow(rgba_image)
                plt.show()

            return rgba_image

    def create_attention_photo(self, og_image: Image, attention_mask_image, save: bool = False, show: bool = False):
        # Ensure both images are in the same mode
        if og_image.mode != 'RGBA':
            og_image = og_image.convert('RGBA')

        # Overlay the second image onto the first image
        # The second image must be the same size as the first image
        og_image.paste(attention_mask_image, (0, 0), attention_mask_image)

        if save:
            og_image.save('image_with_attention.png')
        if show:
            plt.imshow(og_image)
            plt.show()

        return og_image

    def return_attention_map(self, filepath: str, show: bool = False, mask_only: bool = False):
        with torch.inference_mode():
            # I know this is a weird way to do this but it works for now
            og_image = Image.open(filepath)
            (original_w, original_h) = og_image.size

            if show:
                plt.imshow(og_image)
                plt.show()

            img = self.prepare_image(cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))[0]
            w, h = img.shape[1] - img.shape[1] % self.patch_size, img.shape[2] - img.shape[2] % self.patch_size
            img = img[:, :w, :h]

            w_featmap = img.shape[-2] // self.patch_size
            h_featmap = img.shape[-1] // self.patch_size

            img = img.unsqueeze(0)
            img = img.to(self.device)
            attention = self.model.get_last_self_attention(img.to(self.device))
            del img, w, h
            torch.cuda.empty_cache()
            
            number_of_heads = attention.shape[1]

            # attention tokens are packed in after the first token; the spatial tokens follow
            attention = attention[0, :, 0, 1 + self.n_register_tokens:].reshape(number_of_heads, -1)

            # resolution of attention from transformer tokens
            attention = attention.reshape(number_of_heads, w_featmap, h_featmap)
            
            # upscale to higher resolution closer to original image
            attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=self.patch_size, mode = "nearest")[0].cpu()

            # sum all attention across the 12 different heads, to get one map of attention across entire image
            attention = torch.sum(attention, dim=0)

            # interpolate attention map back into original image dimensions
            attention = nn.functional.interpolate(attention.unsqueeze(0).unsqueeze(0), size=(original_h, original_w), mode='bilinear', align_corners=False)
            del original_h, original_w, w_featmap, h_featmap, number_of_heads
            torch.cuda.empty_cache()
            
            attention = attention.squeeze()
            image_metric = attention.numpy()
            del attention
            torch.cuda.empty_cache()

            attention_mask = self.create_attention_mask(image_metric, show=show)
            del image_metric
            torch.cuda.empty_cache()

            if mask_only:
                return attention_mask
            
            photo_with_attention = self.create_attention_photo(og_image, attention_mask, show=show)
            del og_image
            torch.cuda.empty_cache()

            return attention_mask, photo_with_attention

    def draw_lines(self, image1, image2, origin, distance_threshold=0.2, max_lines=10, n_neighbors=1):
        # Read images in RGB
        img1 = cv2.cvtColor(cv2.imread(image1, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(image2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Compute common dimensions (minimum size) and resize both images
        common_height = (img1.shape[0] + img2.shape[0]) // 2
        common_width = (img1.shape[1] + img2.shape[1]) // 2
        img1 = cv2.resize(img1, (common_width, common_height))
        img2 = cv2.resize(img2, (common_width, common_height))
        
        # Preprocess images with model (get grid and resize_scale)
        _, grid_size1, resize_scale1 = self.prepare_image(img1)
        _, grid_size2, resize_scale2 = self.prepare_image(img2)

        # Extract features without pooling
        features1 = self.extract_features(img1, pooling=False)
        features2 = self.extract_features(img2, pooling=False)
        
        # Use KNN to find matches
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(features1)
        distances, matches = knn.kneighbors(features2)

        # Normalize distances
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-8)
        
        # Filter the matches based on normalized distance
        selected_matches = []
        for idx2, (dist_row, match_row) in enumerate(zip(distances, matches)):
            for dist, idx1 in zip(dist_row, match_row):
                if dist < distance_threshold:
                    selected_matches.append((dist, idx1, idx2))

        # Prepare the visualization
        fig = plt.figure(figsize=(20, 10))  
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.imshow(img1)
        ax1.axis("off")
        ax1.set_title("Final submission")
        ax2.imshow(img2)
        ax2.axis("off")
        ax2.set_title(f"Closest inspiration - {origin}")

        if len(selected_matches) != 0:
            # Sort and limit number of lines
            selected_matches = sorted(selected_matches, key=lambda x: x[0])[:max_lines]

            for dist, idx1, idx2 in selected_matches:
                # Get pixel coordinates for matched tokens
                row, col = self.idx_to_source_position(idx1, grid_size1, resize_scale1)
                xyA = (col, row)
                row, col = self.idx_to_source_position(idx2, grid_size2, resize_scale2)
                xyB = (col, row)

                similarity = 1 - (dist / distance_threshold) # convert distance to similarity
                color = cm.viridis(similarity)
                linewidth = 3 # fixed linewidth
                
                con = ConnectionPatch(xyA=xyB, xyB=xyA, coordsA="data", coordsB="data",
                                    axesA=ax2, axesB=ax1, color=color, linewidth=linewidth)
                ax2.add_artist(con)

        plt.show()


    def draw_attention(self, pic1, pic2):
        for picture in [pic1, pic2]:
            attn_mask, attn_photo = self.return_attention_map(picture, show=False)
            fig = plt.figure(figsize=(20, 10))
                    
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            
            ax1.imshow(cv2.cvtColor(cv2.imread(picture, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            ax1.set_title("Photo")
            ax1.axis("off")
                    
            ax2.imshow(attn_mask)
            ax2.set_title("Attention Mask")
            ax2.axis("off")
                    
            ax3.imshow(attn_photo)
            ax3.set_title("Masked Photo")
            ax3.axis("off")
                    
            plt.tight_layout()
            plt.show()