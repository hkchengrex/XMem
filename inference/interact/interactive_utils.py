# Modifed from https://github.com/seoungwugoh/ivs-demo

import numpy as np

from scipy.ndimage.morphology import binary_dilation

import torch
import torch.nn.functional as F
from dataset.range_transform import im_normalization

def image_to_torch(frame: np.ndarray, device='cuda'):
    # frame: H*W*3 numpy array
    frame = frame.transpose(2, 0, 1)
    frame = torch.from_numpy(frame).float().to(device)/255
    frame_norm = im_normalization(frame)
    return frame_norm, frame

def torch_prob_to_numpy_mask(prob):
    mask = torch.argmax(prob, dim=0)
    mask = mask.cpu().numpy().astype(np.uint8)
    return mask

def index_numpy_to_one_hot_torch(mask, num_classes):
    mask = torch.from_numpy(mask).long()
    return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()

color_map = [
    [0, 0, 0], 
    [255, 50, 50], 
    [50, 255, 50], 
    [50, 50, 255], 
    [255, 50, 255], 
    [50, 255, 255], 
    [255, 255, 50], 
]

color_map_np = np.array(color_map)
if torch.cuda.is_available():
    color_map_torch = torch.from_numpy(color_map_np).cuda() / 255

def overlay_davis(image, mask, alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    # countours = binary_dilation(binary_mask) ^ binary_mask
    # im_overlay[countours,:] = 0
    return im_overlay.astype(image.dtype)

def overlay_davis_fade(image, mask, alpha=0.5):
    im_overlay = image.copy()

    colored_mask = color_map_np[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    # countours = binary_dilation(binary_mask) ^ binary_mask
    # im_overlay[countours,:] = 0
    im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)

def overlay_davis_torch(image, mask, alpha=0.5):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.argmax(mask, dim=0)

    colored_mask = color_map_torch[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    # countours = binary_dilation(binary_mask) ^ binary_mask
    # im_overlay[countours,:] = 0

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay

def overlay_davis_fade_torch(image, mask, alpha=0.5):
    # Changes the image in-place to avoid copying
    image = image.permute(1, 2, 0)
    im_overlay = image
    mask = torch.argmax(mask, dim=0)

    colored_mask = color_map_torch[mask]
    foreground = image*alpha + (1-alpha)*colored_mask
    binary_mask = (mask > 0)
    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]
    # countours = binary_dilation(binary_mask) ^ binary_mask
    # im_overlay[countours,:] = 0
    im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6

    im_overlay = (im_overlay*255).cpu().numpy()
    im_overlay = im_overlay.astype(np.uint8)

    return im_overlay
