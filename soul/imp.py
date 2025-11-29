"""
Imp - Single-layer convolution soul implementation.

Named after a lesser demon, this soul uses a single layer of convolution
for simpler, more immediate transformations.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Imp(Soul):
    """
    Simple single-layer convolution processor.
    
    A simpler variation that applies just one convolution layer,
    useful for faster processing or different visual effects.
    """
    
    def __init__(self, kernel_size=7, drift_magnitude=0.002, momentum=0.9, 
                 nonlinearity_scale=0.5, device=None):
        """
        Initialize single-layer convolution processor.
        
        Args:
            kernel_size: Size of convolution kernel (default: 7)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            nonlinearity_scale: Scale factor for final tanh (default: 0.5)
            device: PyTorch device (cuda or cpu)
        """
        self.kernel_size = kernel_size
        self.nonlinearity_scale = nonlinearity_scale
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                        momentum=momentum, device=device)
    
    def _initialize_kernels(self):
        """Generate a single random convolution kernel."""
        kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
        
        # Kill DC mode
        kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
        
        # Normalize
        kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
        
        return [kernel]
    
    def apply(self, image, residual_alpha=0.2):
        """
        Apply single convolution layer with residual connection.
        
        Args:
            image: Input image tensor (C, H, W) on any device
            residual_alpha: Blending factor for residual connection
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        pad_size = self.padding
        kernel = self.kernels[0].to(self.device)
        
        # Move image to device and add batch dimension
        img_batch = image.to(self.device).unsqueeze(0)
        
        # Apply circular padding
        img_padded = F.pad(img_batch, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        
        # Apply convolution
        conv_result = F.conv2d(img_padded, kernel, padding=0).squeeze(0)
        
        # Apply non-linearity
        conv_result = torch.tanh(conv_result)
        
        # Residual blending
        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * conv_result
        
        # Normalize
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        # Keep on device for efficiency
        return result
    
    def get_soul_sliders(self):
        """Return Imp-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1
            }
        ]

