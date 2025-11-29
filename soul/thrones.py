"""
Thrones - Multi-resolution cascade soul with flow fields.

Thrones sit at the intersection of scales: processes the image as a
pyramid of resolutions with inter-scale flow fields that create smooth,
liquid transformations. Like watching reality melt and reform through
nested lenses.
"""

import torch
import torch.nn.functional as F
import math

from .base import Soul


class Thrones(Soul):
    """
    Multi-resolution processor with optical flow-like motion.

    Creates a scale pyramid, processes each level with different kernels,
    and uses flow fields to create smooth motion between scales during
    reconstruction. Results in liquid, melting transformations.
    """

    def __init__(self, kernel_size=5, num_scales=3, drift_magnitude=0.003,
                 momentum=0.7, flow_strength=0.15, scale_blend=0.5,
                 flow_frequency=1.5, temporal_smooth=0.3, 
                 nonlinearity_scale=0.7, device=None):
        """
        Args:
            kernel_size: Size of convolution kernels.
            num_scales: Number of pyramid levels (default: 3).
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            flow_strength: Strength of inter-scale flow fields (default: 0.15).
            scale_blend: How much to blend between scales (default: 0.5).
            flow_frequency: Spatial frequency of flow patterns (default: 1.5).
            temporal_smooth: Temporal smoothing of flow evolution (default: 0.3).
            nonlinearity_scale: Scale factor for final tanh (default: 0.7).
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.flow_strength = flow_strength
        self.scale_blend = scale_blend
        self.flow_frequency = flow_frequency
        self.temporal_smooth = temporal_smooth
        self.nonlinearity_scale = nonlinearity_scale
        
        # Flow evolution state
        self.flow_phase = 0.0
        self.prev_pyramid = None  # For temporal smoothing
        
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """
        Create kernels for each scale level.
        
        Each scale gets its own kernel with different characteristics:
        - Fine scales: sharper, edge-enhancing
        - Coarse scales: smoother, structure-preserving
        """
        kernels = []
        
        for i in range(self.num_scales):
            # Scale-dependent kernel design
            kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
            
            # Create spatial modulation based on scale
            y, x = torch.meshgrid(
                torch.arange(self.kernel_size, dtype=torch.float32), 
                torch.arange(self.kernel_size, dtype=torch.float32), 
                indexing='ij'
            )
            center = self.kernel_size / 2
            dist = torch.sqrt((x - center)**2 + (y - center)**2)
            
            # Fine scales: amplify high frequencies (sharper)
            # Coarse scales: suppress high frequencies (smoother)
            scale_factor = i / max(1, self.num_scales - 1)  # 0 to 1
            frequency_weight = torch.exp(-scale_factor * dist / (self.kernel_size / 2))
            
            kernel = kernel * frequency_weight.unsqueeze(0).unsqueeze(0)
            
            # Normalize to zero mean and unit RMS
            kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
            kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
            
            kernels.append(kernel)
        
        return kernels

    def _generate_flow_field(self, height, width):
        """
        Generate smooth flow field for inter-scale warping.
        
        Creates a time-evolving vector field that determines how pixels
        flow between scales. Like optical flow but artistically driven.
        
        Returns:
            Flow field tensor of shape (2, H, W) for (dx, dy)
        """
        # Normalized coordinates
        y = torch.linspace(0, 1, height, device=self.device)
        x = torch.linspace(0, 1, width, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Create smooth, organic flow patterns
        phase = self.flow_phase
        freq = self.flow_frequency * 2 * math.pi
        
        # X-component of flow (horizontal)
        flow_x = torch.sin(freq * yy + phase) * torch.cos(freq * 0.7 * xx - phase * 0.8)
        flow_x += 0.5 * torch.cos(freq * 1.3 * (xx + yy) + phase * 1.5)
        
        # Y-component of flow (vertical)
        flow_y = torch.cos(freq * xx - phase * 1.2) * torch.sin(freq * 0.8 * yy + phase)
        flow_y += 0.5 * torch.sin(freq * 1.4 * (xx - yy) - phase * 1.3)
        
        # Add vortex-like structures
        cx, cy = 0.5, 0.5
        dx, dy = xx - cx, yy - cy
        r = torch.sqrt(dx**2 + dy**2 + 1e-6)
        theta = torch.atan2(dy, dx)
        
        # Rotating flow component
        vortex_strength = torch.exp(-r * 3) * math.sin(phase * 2)
        flow_x += vortex_strength * (-torch.sin(theta))
        flow_y += vortex_strength * torch.cos(theta)
        
        # Scale to pixel units
        flow_x = flow_x * self.flow_strength * width / 10
        flow_y = flow_y * self.flow_strength * height / 10
        
        # Stack into (2, H, W)
        flow = torch.stack([flow_x, flow_y], dim=0)
        
        return flow

    def _warp_image(self, image, flow):
        """
        Warp an image according to a flow field.
        
        Args:
            image: (1, C, H, W) image tensor
            flow: (2, H, W) flow field (dx, dy)
            
        Returns:
            Warped image (1, C, H, W)
        """
        _, _, H, W = image.shape
        
        # Create sampling grid
        y = torch.arange(H, dtype=torch.float32, device=self.device)
        x = torch.arange(W, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Apply flow
        new_x = xx + flow[0]
        new_y = yy + flow[1]
        
        # Normalize to [-1, 1] for grid_sample
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0
        
        # Stack and reshape for grid_sample
        grid = torch.stack([new_x, new_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        
        # Sample with bilinear interpolation
        # Note: using 'zeros' padding mode for MPS compatibility
        warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return warped

    def apply(self, image, residual_alpha=0.3):
        """
        Apply multi-resolution cascade with flow-based reconstruction.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new transformation.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        img = image.to(self.device)
        _, orig_h, orig_w = img.shape
        img_batch = img.unsqueeze(0)
        
        # Evolve flow phase
        self.flow_phase += 0.02
        
        # Build pyramid: downsample to multiple scales
        pyramid = [img_batch]
        for i in range(1, self.num_scales):
            # Downsample by factor of 2
            downsampled = F.avg_pool2d(pyramid[-1], kernel_size=2, stride=2)
            pyramid.append(downsampled)
        
        # Process each scale with its kernel
        processed_pyramid = []
        pad_size = self.padding
        
        for scale_idx, (scale_img, kernel) in enumerate(zip(pyramid, self.kernels)):
            # Apply circular padding
            padded = F.pad(scale_img, (pad_size, pad_size, pad_size, pad_size), mode='circular')
            
            # Convolve
            conv_result = F.conv2d(padded, kernel.to(self.device), padding=0)
            
            # Apply nonlinearity
            conv_result = torch.tanh(conv_result)
            
            processed_pyramid.append(conv_result)
        
        # Reconstruct from coarse to fine with flow-based warping
        result = processed_pyramid[-1]  # Start with coarsest scale
        
        for scale_idx in range(self.num_scales - 2, -1, -1):
            # Upsample current result to next finer scale
            target_h, target_w = processed_pyramid[scale_idx].shape[2:]
            upsampled = F.interpolate(result, size=(target_h, target_w), 
                                     mode='bilinear', align_corners=True)
            
            # Generate flow field for this scale
            flow = self._generate_flow_field(target_h, target_w)
            
            # Warp the upsampled result
            warped = self._warp_image(upsampled, flow * (scale_idx + 1) / self.num_scales)
            
            # Blend with processed version at this scale
            result = (1 - self.scale_blend) * warped + self.scale_blend * processed_pyramid[scale_idx]
            result = torch.tanh(result)
        
        # Temporal smoothing with previous frame
        if self.prev_pyramid is not None and self.temporal_smooth > 0:
            prev = self.prev_pyramid.to(self.device)
            if prev.shape == result.shape:
                result = (1 - self.temporal_smooth) * result + self.temporal_smooth * prev
        
        self.prev_pyramid = result.detach().cpu()
        
        # Final resize to original dimensions if needed
        if result.shape[2:] != (orig_h, orig_w):
            result = F.interpolate(result, size=(orig_h, orig_w), 
                                  mode='bilinear', align_corners=True)
        
        # Residual blending with input
        result = (1 - residual_alpha) * img_batch + residual_alpha * result
        
        # Normalize
        result = result.squeeze(0)
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        return result
    
    def get_soul_sliders(self):
        """Return Thrones-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Flow Strength",
                "value_attr": "flow_strength",
                "min_value": 0.0,
                "max_value": 0.5
            },
            {
                "label": "Scale Blend",
                "value_attr": "scale_blend",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Flow Frequency",
                "value_attr": "flow_frequency",
                "min_value": 0.5,
                "max_value": 4.0
            },
            {
                "label": "Temporal Smooth",
                "value_attr": "temporal_smooth",
                "min_value": 0.0,
                "max_value": 0.7
            }
        ]

