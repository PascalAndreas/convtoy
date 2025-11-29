"""
Nephilim - Chaotic multi-zone soul with spatial turbulence.

Half-angel, half-demon: combines ordered convolutions with chaotic
spatial modulation. Different regions process differently based on
evolving turbulence fields, creating dynamic, non-uniform behavior.
"""

import torch
import torch.nn.functional as F
import math

from .base import Soul


class Nephilim(Soul):
    """
    Spatially chaotic multi-kernel processor with turbulent zones.

    Maintains multiple kernels with different "personalities" and uses
    evolving turbulence fields to select and blend between them spatially.
    Creates weather-like patterns of different processing behaviors that
    drift across the image.
    """

    def __init__(self, kernel_size=7, num_kernels=4, drift_magnitude=0.002,
                 momentum=0.75, turbulence_scale=0.3, chaos_speed=0.05,
                 zone_frequency=2.0, feedback_strength=0.15, 
                 contrast_boost=0.7, nonlinearity_scale=0.8, device=None):
        """
        Args:
            kernel_size: Size of convolution kernels.
            num_kernels: Number of different kernel personalities (default: 4).
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            turbulence_scale: Intensity of spatial turbulence (default: 0.3).
            chaos_speed: Speed of turbulence evolution (default: 0.05).
            zone_frequency: Vibrational mode frequency (like cymatics) (default: 2.0).
            feedback_strength: Strength of temporal feedback (default: 0.15).
            contrast_boost: Boost for high-contrast areas (default: 0.7).
            nonlinearity_scale: Scale factor for final tanh (default: 0.8).
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.turbulence_scale = turbulence_scale
        self.chaos_speed = chaos_speed
        self.zone_frequency = zone_frequency
        self.feedback_strength = feedback_strength
        self.contrast_boost = contrast_boost
        self.nonlinearity_scale = nonlinearity_scale
        
        # Turbulence state (phase offsets for evolving patterns)
        self.turbulence_phase = 0.0
        self.prev_output = None  # For feedback
        
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """
        Create a bank of kernels with different personalities:
        - Smooth (low-pass)
        - Sharp (high-pass)
        - Directional (oriented edges)
        - Swirly (rotation-like)
        """
        kernels = []
        
        for i in range(self.num_kernels):
            if i == 0:
                # Smooth/blurring kernel
                kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
                kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
                # Make it more smooth by reducing high frequencies
                kernel = kernel * torch.exp(-0.3 * torch.randn_like(kernel)**2)
                
            elif i == 1:
                # Sharp/edge-detecting kernel
                kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
                center = self.kernel_size // 2
                # Amplify center vs surround (Laplacian-like)
                y, x = torch.meshgrid(torch.arange(self.kernel_size), torch.arange(self.kernel_size), indexing='ij')
                dist = torch.sqrt((x - center)**2 + (y - center)**2)
                weight = 1.0 - 0.8 * torch.exp(-dist**2 / 2)
                kernel = kernel * weight.unsqueeze(0).unsqueeze(0)
                
            elif i == 2:
                # Directional kernel (oriented features)
                kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
                y, x = torch.meshgrid(torch.arange(self.kernel_size), torch.arange(self.kernel_size), indexing='ij')
                angle = math.pi * (i / self.num_kernels)
                oriented = math.cos(angle) * (x - self.kernel_size/2) + math.sin(angle) * (y - self.kernel_size/2)
                kernel = kernel * (1 + 0.5 * oriented.unsqueeze(0).unsqueeze(0))
                
            else:
                # Swirly kernel (rotation-like)
                kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
                y, x = torch.meshgrid(torch.arange(self.kernel_size), torch.arange(self.kernel_size), indexing='ij')
                cy, cx = self.kernel_size / 2, self.kernel_size / 2
                dy, dx = y - cy, x - cx
                angle = torch.atan2(dy, dx)
                kernel = kernel * torch.sin(angle * 2).unsqueeze(0).unsqueeze(0)
            
            # Normalize to zero mean and unit RMS
            kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
            kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
            kernels.append(kernel)
        
        return kernels

    def _generate_turbulence_field(self, height, width):
        """
        Generate evolving turbulence field using Chladni-like patterns.
        
        Creates standing wave patterns similar to cymatics (vibrating plates
        with sand), where different vibrational modes create organic nodal
        patterns that determine which kernel is used where.
        
        Returns:
            Turbulence field tensor of shape (1, num_kernels, H, W)
        """
        # Normalized coordinate grids [-1, 1]
        y = torch.linspace(-1, 1, height, device=self.device)
        x = torch.linspace(-1, 1, width, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Polar coordinates for radial modes
        r = torch.sqrt(xx**2 + yy**2)
        theta = torch.atan2(yy, xx)
        
        turbulence_maps = []
        
        for k in range(self.num_kernels):
            # Each kernel gets its own Chladni-like pattern
            # Use different mode numbers (m, n) for each kernel
            
            # Mode parameters that evolve over time
            phase = self.turbulence_phase + k * math.pi / self.num_kernels
            
            # Chladni patterns are standing waves: cos(m*theta) * J_n(freq*r)
            # We'll approximate with simpler functions for efficiency
            
            # Angular mode number (creates symmetry)
            m = k + 2  # Angular symmetry order
            
            # Radial component - multiple frequencies create nodal circles
            radial_freq = self.zone_frequency * (k + 1)
            radial_component = torch.cos(radial_freq * math.pi * r + phase)
            
            # Angular component - creates symmetric petals/lobes
            angular_component = torch.cos(m * theta + phase * 0.3)
            
            # Combine radial and angular (classic Chladni pattern)
            chladni = radial_component * angular_component
            
            # Add a second mode at different frequency for complexity
            m2 = (k + 1) * 2
            radial_freq2 = self.zone_frequency * 1.6 * (k + 1)
            chladni2 = torch.cos(radial_freq2 * math.pi * r + phase * 1.4) * torch.sin(m2 * theta - phase * 0.5)
            
            # Superposition of two modes (like multiple frequencies vibrating the plate)
            field = chladni + 0.5 * chladni2
            
            # Add some drift/wobble based on time
            drift_x = 0.1 * math.sin(phase * 0.7)
            drift_y = 0.1 * math.cos(phase * 0.9)
            xx_drift = xx - drift_x
            yy_drift = yy - drift_y
            r_drift = torch.sqrt(xx_drift**2 + yy_drift**2)
            theta_drift = torch.atan2(yy_drift, xx_drift)
            
            # Add a third, slower-evolving mode for organic variation
            wobble = 0.3 * torch.sin(self.zone_frequency * 0.8 * math.pi * r_drift) * torch.cos((k+3) * theta_drift + phase * 0.2)
            field = field + wobble
            
            turbulence_maps.append(field)
        
        # Stack and normalize to weights
        turbulence = torch.stack(turbulence_maps, dim=0).unsqueeze(0)  # (1, K, H, W)
        
        # Apply turbulence scale and softmax for smooth blending
        turbulence = turbulence * self.turbulence_scale
        turbulence = F.softmax(turbulence, dim=1)  # Sum to 1 across kernels
        
        return turbulence

    def apply(self, image, residual_alpha=0.25):
        """
        Apply spatially-varying multi-kernel convolution with turbulent zones.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new transformation.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        img = image.to(self.device)
        _, height, width = img.shape
        
        # Evolve turbulence phase
        self.turbulence_phase += self.chaos_speed
        
        # Generate spatial turbulence field
        turbulence = self._generate_turbulence_field(height, width)  # (1, K, H, W)
        
        # Apply temporal feedback if available
        if self.prev_output is not None and self.feedback_strength > 0:
            feedback = self.prev_output.to(self.device)
            # Use feedback as additional modulation
            feedback_intensity = feedback.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
            feedback_mod = 1.0 + self.feedback_strength * torch.tanh(feedback_intensity)
            turbulence = turbulence * feedback_mod
            # Re-normalize
            turbulence = turbulence / turbulence.sum(dim=1, keepdim=True)
        
        # Compute local contrast for adaptive processing
        img_gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
        local_std = F.avg_pool2d(
            F.pad(img_gray**2, (2, 2, 2, 2), mode='replicate'), 
            kernel_size=5, stride=1
        ) - F.avg_pool2d(
            F.pad(img_gray, (2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )**2
        local_std = torch.sqrt(torch.clamp(local_std, min=0) + 1e-6)
        contrast_boost = 1.0 + self.contrast_boost * local_std / (local_std.mean() + 1e-6)
        
        # Apply each kernel and blend according to turbulence field
        img_batch = img.unsqueeze(0)
        accumulated = torch.zeros_like(img_batch)
        
        pad_size = self.padding
        img_padded = F.pad(img_batch, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        
        for k, kernel in enumerate(self.kernels):
            # Apply convolution
            conv_result = F.conv2d(img_padded, kernel.to(self.device), padding=0)
            
            # Apply nonlinearity
            conv_result = torch.tanh(conv_result)
            
            # Weight by turbulence field
            weight = turbulence[:, k:k+1, :, :]  # (1, 1, H, W)
            accumulated += weight * conv_result
        
        # Apply contrast boost spatially
        accumulated = accumulated * contrast_boost
        
        # Residual blending
        result = (1 - residual_alpha) * img_batch + residual_alpha * accumulated
        
        # Normalize
        result = result.squeeze(0)
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        # Store for feedback
        self.prev_output = result.detach().cpu()
        
        return result
    
    def get_soul_sliders(self):
        """Return Nephilim-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Turbulence Scale",
                "value_attr": "turbulence_scale",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Chaos Speed",
                "value_attr": "chaos_speed",
                "min_value": 0.0,
                "max_value": 0.2
            },
            {
                "label": "Cymatic Frequency",
                "value_attr": "zone_frequency",
                "min_value": 0.5,
                "max_value": 5.0
            },
            {
                "label": "Feedback",
                "value_attr": "feedback_strength",
                "min_value": 0.0,
                "max_value": 0.5
            },
            {
                "label": "Contrast Boost",
                "value_attr": "contrast_boost",
                "min_value": 0.0,
                "max_value": 1.5
            }
        ]

