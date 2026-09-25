import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

class SemanticSegmentationTarget:
    """
    Tells Grad-CAM to focus only on the gradients that contribute 
    to the positively predicted lesion mask.
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, model_output):
        return (model_output * self.mask).sum()

def generate_gradcam(model, image_tensor, original_image_rgb, device='cpu'):
    model.eval()
    
    # Target the deepest convolutional layer in the decoder.
    target_layers = [model.dec4.block[-1]] 
    
    # We must enable gradients for Grad-CAM to work
    image_tensor = image_tensor.to(device).requires_grad_(True)
    
    # 1. Get the model's prediction first to know where the lesion is
    with torch.no_grad():
        raw_output = model(image_tensor)
        pred_mask = (torch.sigmoid(raw_output) > 0.5).float()
    
    # 2. Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 3. Target the specific predicted mask
    targets = [SemanticSegmentationTarget(pred_mask)]
    
    # 4. Generate the heatmap
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    
    # 5. Resize heatmap to match the original image dimensions
    orig_h, orig_w = original_image_rgb.shape[:2]
    grayscale_cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h))
    
    # 6. Overlay on original image
    img_normalized = original_image_rgb.astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_normalized, grayscale_cam_resized, use_rgb=True)
    
    return overlay, grayscale_cam_resized