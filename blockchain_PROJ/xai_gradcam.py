import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
from model_inference import load_model, preprocess_image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        model_output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = model_output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = model_output[0, class_idx]
        class_loss.backward()
        
        # Generate CAM
        gradients = self.gradients.data
        activations = self.activations.data
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()


def get_target_layer(model):
    """Get the final convolutional layer for Grad-CAM."""
    from model_inference import DummyXceptionModel
    if isinstance(model, DummyXceptionModel):
        return model.conv2
    else:
        # For the Xception model, use conv4 as the final conv layer
        return model.conv4


def generate_gradcam_heatmap(image_path, model_path="models/xception_gan_augmented.pth"):
    """Generate Grad-CAM heatmap for an input image."""
    # Load model
    model, device = load_model(model_path)
    target_layer = get_target_layer(model)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Preprocess image
    input_tensor = preprocess_image(image_path).to(device)
    
    # Generate CAM
    cam = gradcam.generate_cam(input_tensor)
    
    return cam


def overlay_heatmap_on_image(image_path, cam, alpha=0.6):
    """Overlay Grad-CAM heatmap on original image."""
    # Load original image
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (299, 299))
    
    # Resize CAM to match image dimensions
    cam_resized = cv2.resize(cam, (299, 299))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    overlay = alpha * heatmap + (1 - alpha) * original_image
    overlay = np.uint8(overlay)
    
    return overlay


def save_gradcam_visualization(image_path, model_path="models/xception_gan_augmented.pth"):
    """Generate and save Grad-CAM visualization."""
    # Ensure output directory exists
    os.makedirs("outputs/xai", exist_ok=True)
    
    # Generate Grad-CAM
    cam = generate_gradcam_heatmap(image_path, model_path)
    
    # Create overlay
    overlay = overlay_heatmap_on_image(image_path, cam)
    
    # Generate output filename
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"outputs/xai/{filename}_cam.png"
    
    # Save overlay
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(output_path)
    
    return output_path