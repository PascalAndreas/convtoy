"""
Dominions - Orientation-selective soul with gradient steering.

Inspired by Gabor filters and steerable pyramids: uses orientation-selective
convolutions that respond to local image structure. The image's own gradients
guide which orientations are enhanced, creating flowing directional patterns
that respect and amplify content rather than overwhelming it.
"""

import torch
import torch.nn.functional as F
import math

from .base import Soul


class Dominions(Soul):
    """
    Orientation-selective processor with gradient-based steering.

    Creates banks of orientation-tuned filters (like Gabor wavelets) at
    multiple frequencies. Local image gradients determine which orientations
    are active where, creating adaptive, structure-aware transformations.
    """

    def __init__(self, kernel_size=7, num_orientations=8, num_frequencies=2,
                 drift_magnitude=0.0025, momentum=0.75, gradient_sensitivity=0.6,
                 orientation_coherence=0.4, frequency_spread=2.0, sharpness=0.8,
                 anisotropy=0.7, phase_offset=0.0, nonlinearity_scale=0.65, device=None):
        """
        Args:
            kernel_size: Size of convolution kernels (should be odd).
            num_orientations: Number of orientation channels (default: 8).
            num_frequencies: Number of frequency bands (default: 2).
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            gradient_sensitivity: How much gradients steer orientation (default: 0.6).
            orientation_coherence: Spatial coherence of orientation field (default: 0.4).
            frequency_spread: Ratio between frequency bands (default: 2.0).
            sharpness: Angular selectivity of filters (default: 0.8).
            anisotropy: Directional vs isotropic balance (default: 0.7).
            phase_offset: Phase offset for even/odd symmetry (default: 0.0).
            nonlinearity_scale: Scale factor for final tanh (default: 0.65).
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.num_orientations = num_orientations
        self.num_frequencies = num_frequencies
        self.gradient_sensitivity = gradient_sensitivity
        self.orientation_coherence = orientation_coherence
        self.frequency_spread = frequency_spread
        self.sharpness = sharpness
        self.anisotropy = anisotropy
        self.phase_offset = phase_offset
        self.nonlinearity_scale = nonlinearity_scale
        
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """
        Create Gabor-like orientation-selective kernels.
        
        For each frequency band, creates a set of oriented kernels that
        respond to edges/patterns at different angles.
        """
        kernels = []
        
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(self.kernel_size, dtype=torch.float32),
            torch.arange(self.kernel_size, dtype=torch.float32),
            indexing='ij'
        )
        center = self.kernel_size / 2
        y = y - center
        x = x - center
        
        # For each frequency band
        base_freq = 2.0 * math.pi / self.kernel_size
        for freq_idx in range(self.num_frequencies):
            frequency = base_freq * (self.frequency_spread ** freq_idx)
            
            # For each orientation
            for orient_idx in range(self.num_orientations):
                angle = orient_idx * math.pi / self.num_orientations
                
                # Rotate coordinates
                x_rot = x * math.cos(angle) + y * math.sin(angle)
                y_rot = -x * math.sin(angle) + y * math.cos(angle)
                
                # Gabor-like kernel: Gaussian envelope * oriented sinusoid
                # Anisotropic Gaussian (elongated along orientation)
                sigma_along = self.kernel_size / 4
                sigma_across = sigma_along / (1 + self.anisotropy * 2)
                
                gaussian = torch.exp(
                    -(x_rot**2 / (2 * sigma_along**2) + y_rot**2 / (2 * sigma_across**2))
                )
                
                # Oriented sinusoidal carrier
                sinusoid = torch.cos(frequency * x_rot + self.phase_offset)
                
                # Combine
                gabor = gaussian * sinusoid
                
                # Make it zero-mean
                gabor = gabor - gabor.mean()
                
                # Replicate to 3 input channels, 3 output channels
                kernel = torch.zeros(3, 3, self.kernel_size, self.kernel_size)
                for c in range(3):
                    kernel[c, c] = gabor
                
                # Normalize
                kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
                
                kernels.append(kernel)
        
        return kernels

    def _compute_gradient_field(self, image):
        """
        Compute local gradient orientation and magnitude.
        
        Args:
            image: (1, C, H, W) tensor
            
        Returns:
            orientation: (1, 1, H, W) angle of gradient in radians
            magnitude: (1, 1, H, W) strength of gradient
        """
        # Sobel filters for gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Convert to grayscale
        gray = image.mean(dim=1, keepdim=True)
        
        # Compute gradients
        grad_x = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), sobel_x)
        grad_y = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='replicate'), sobel_y)
        
        # Gradient orientation (perpendicular to edge orientation)
        orientation = torch.atan2(grad_y, grad_x)
        
        # Gradient magnitude
        magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return orientation, magnitude

    def _compute_orientation_weights(self, gradient_orientation, gradient_magnitude, height, width):
        """
        Compute spatial weights for each orientation based on local gradients.
        
        Args:
            gradient_orientation: (1, 1, H, W) angle map
            gradient_magnitude: (1, 1, H, W) magnitude map
            height, width: output dimensions
            
        Returns:
            weights: (1, num_orientations * num_frequencies, H, W)
        """
        # Resize if needed
        if gradient_orientation.shape[2:] != (height, width):
            gradient_orientation = F.interpolate(gradient_orientation, size=(height, width), 
                                                 mode='bilinear', align_corners=True)
            gradient_magnitude = F.interpolate(gradient_magnitude, size=(height, width),
                                               mode='bilinear', align_corners=True)
        
        weights_list = []
        
        for freq_idx in range(self.num_frequencies):
            for orient_idx in range(self.num_orientations):
                # Target orientation for this filter
                target_angle = orient_idx * math.pi / self.num_orientations
                
                # Angular difference between gradient and filter orientation
                angle_diff = gradient_orientation - target_angle
                # Wrap to [-pi, pi]
                angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                
                # Von Mises-like angular selectivity (concentrated around 0)
                angular_response = torch.exp(self.sharpness * (torch.cos(angle_diff) - 1))
                
                # Modulate by gradient magnitude (stronger gradients = stronger response)
                magnitude_norm = gradient_magnitude / (gradient_magnitude.mean() + 1e-6)
                response = angular_response * (1 + self.gradient_sensitivity * magnitude_norm)
                
                weights_list.append(response)
        
        weights = torch.cat(weights_list, dim=1)  # (1, num_orientations * num_frequencies, H, W)
        
        # Apply spatial coherence (smooth the weight field)
        if self.orientation_coherence > 0:
            kernel_size = 5
            blur = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size ** 2)
            weights_smooth = F.conv2d(
                F.pad(weights, (2, 2, 2, 2), mode='replicate'),
                blur.repeat(weights.shape[1], 1, 1, 1),
                groups=weights.shape[1]
            )
            weights = (1 - self.orientation_coherence) * weights + self.orientation_coherence * weights_smooth
        
        # Normalize weights to sum to approximately 1
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-6)
        
        return weights

    def apply(self, image, residual_alpha=0.25):
        """
        Apply orientation-selective convolutions with gradient steering.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new transformation.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        img = image.to(self.device)
        _, height, width = img.shape
        img_batch = img.unsqueeze(0)
        
        # Compute gradient field
        grad_orientation, grad_magnitude = self._compute_gradient_field(img_batch)
        
        # Compute orientation weights
        weights = self._compute_orientation_weights(grad_orientation, grad_magnitude, height, width)
        
        # Apply all oriented convolutions
        pad_size = self.padding
        img_padded = F.pad(img_batch, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        
        responses = []
        for kernel in self.kernels:
            response = F.conv2d(img_padded, kernel.to(self.device), padding=0)
            response = torch.tanh(response)
            responses.append(response)
        
        # Stack all responses
        responses_stacked = torch.stack(responses, dim=1)  # (1, num_kernels, C, H, W)
        
        # Weight and sum across orientations
        weights_expanded = weights.unsqueeze(2)  # (1, num_kernels, 1, H, W)
        weighted_sum = (responses_stacked * weights_expanded).sum(dim=1)  # (1, C, H, W)
        
        # Residual blending
        result = (1 - residual_alpha) * img_batch + residual_alpha * weighted_sum
        
        # Normalize
        result = result.squeeze(0)
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        return result
    
    def get_soul_sliders(self):
        """Return Dominions-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Gradient Sensitivity",
                "value_attr": "gradient_sensitivity",
                "min_value": 0.0,
                "max_value": 1.5
            },
            {
                "label": "Orientation Coherence",
                "value_attr": "orientation_coherence",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Frequency Spread",
                "value_attr": "frequency_spread",
                "min_value": 1.0,
                "max_value": 4.0
            },
            {
                "label": "Sharpness",
                "value_attr": "sharpness",
                "min_value": 0.1,
                "max_value": 2.0
            },
            {
                "label": "Anisotropy",
                "value_attr": "anisotropy",
                "min_value": 0.0,
                "max_value": 1.5
            },
            {
                "label": "Phase Offset",
                "value_attr": "phase_offset",
                "min_value": -3.14159,
                "max_value": 3.14159
            }
        ]

