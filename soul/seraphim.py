"""
Seraphim - Multi-layer convolution soul implementation.

Named after the highest order of angels, this soul uses multiple layers
of convolution to create complex, heavenly transformations.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Seraphim(Soul):
    """
    Multi-layer convolution processor with residual connections.
    
    Applies multiple sequential convolution layers with tanh nonlinearity
    and residual blending to create complex spatial transformations.
    """
    
    def __init__(self, kernel_size=7, num_layers=5, drift_magnitude=0.002, 
                 momentum=0.7, nonlinearity_scale=0.5, device=None):
        """
        Initialize multi-layer convolution processor.
        
        Args:
            kernel_size: Size of convolution kernels (default: 7)
            num_layers: Number of sequential convolution layers (default: 5)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            nonlinearity_scale: Scale factor for final tanh (0-1, default: 0.5)
            device: PyTorch device (cuda or cpu)
        """
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.nonlinearity_scale = nonlinearity_scale
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                        momentum=momentum, device=device)
    
    def _initialize_kernels(self):
        """Generate random convolution kernels for each layer."""
        kernels = []
        for _ in range(self.num_layers):
            # Standard spatial sampling
            kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
            
            # CRITICAL: Kill DC mode - make each kernel zero-mean
            # This prevents "single color" collapse
            kernel = kernel - kernel.mean(dim=(-1, -2), keepdim=True)
            
            # Normalize to control spectral radius (prevent explosion)
            kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
            
            kernels.append(kernel)
        return kernels
    
    def apply(self, image, residual_alpha=0.2):
        """
        Apply multiple convolution layers with residual updates and stable normalization.
        
        Args:
            image: Input image tensor (C, H, W)
            residual_alpha: How much of new conv to blend in (0=no change, 1=full replacement)
            
        Returns:
            Processed image tensor (C, H, W)
        """
        result = image
        pad_size = self.padding
        
        # Apply each layer sequentially
        for kernel in self.kernels:
            # Add batch dimension
            img_batch = result.unsqueeze(0).to(self.device)
            kernel = kernel.to(self.device)
            
            # Apply circular padding so edges wrap around
            img_padded = F.pad(img_batch, (pad_size, pad_size, pad_size, pad_size), mode='circular')
            
            # Apply 3-channel convolution (allows color channel mixing)
            conv_result = F.conv2d(img_padded, kernel, padding=0).squeeze(0).cpu()
            
            # Apply non-linearity after each convolution
            conv_result = torch.tanh(conv_result)
            
            # Residual update (blend old and new) - prevents washout
            result = (1 - residual_alpha) * result + residual_alpha * conv_result
        
        # Standardize (keep dynamics stable, not min-max)
        # This maintains contrast structure without brutal rescaling
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)  # Soft clamp
        
        return result
    
    def get_sliders(self):
        """Return Seraphim-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1
            }
        ]

