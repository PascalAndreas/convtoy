"""
Ophanim - Multi-scale depthwise swirl soul.

Wheels within wheels: mixes shallow color weaving with dilated,
channel-wise spirals to generate looping filigree without a heavy model.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Ophanim(Soul):
    """
    Multi-scale depthwise swirl with gentle color weaving.

    Uses a 1x1 color pre-mix, a stack of dilated depthwise filters, and a
    final color fuse to spin different spatial frequencies against each other.
    """

    def __init__(self, kernel_size=5, dilations=(1, 2), drift_magnitude=0.0015,
                 momentum=0.8, nonlinearity_scale=0.6, spiral_intensity=0.25, device=None):
        """
        Args:
            kernel_size: Base kernel size for the depthwise filters.
            dilations: Iterable of dilation factors for the depthwise stack.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            nonlinearity_scale: Scale factor for final tanh (default: 0.6)
            spiral_intensity: Intensity of spiral phase field (default: 0.25)
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.nonlinearity_scale = nonlinearity_scale
        self.spiral_intensity = spiral_intensity
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create pre-mix, depthwise stack, and post-mix kernels."""
        kernels = []

        # Color pre-mix (1x1, light-handed so colors weave before swirling)
        pre = torch.randn(3, 3, 1, 1)
        pre = pre - pre.mean(dim=(0, 2, 3), keepdim=True)
        pre = pre / (pre.std() + 1e-6)
        kernels.append(pre)

        # Dilated depthwise stack (one per dilation)
        for _ in self.dilations:
            depthwise = torch.randn(3, 1, self.kernel_size, self.kernel_size)
            depthwise = depthwise - depthwise.mean(dim=(-1, -2), keepdim=True)
            depthwise = depthwise / (depthwise.std() * (self.kernel_size ** 2) + 1e-6)
            kernels.append(depthwise)

        # Color fuse (1x1) after the swirl is assembled
        post = torch.randn(3, 3, 1, 1)
        post = post - post.mean(dim=(0, 2, 3), keepdim=True)
        post = post / (post.std() + 1e-6)
        kernels.append(post)
        return kernels

    def apply(self, image, residual_alpha=0.25):
        """
        Apply swirling multi-scale depthwise convolutions with color weaving.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new swirl versus the old frame.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        # Stage 1: color pre-mix (keeps hue motion alive)
        x = image.to(self.device).unsqueeze(0)
        pre = F.conv2d(x, self.kernels[0].to(self.device), padding=0)
        pre = torch.tanh(pre)

        # Stage 2: depthwise swirls at different dilations
        swirls = []
        for dilation, kernel in zip(self.dilations, self.kernels[1:-1]):
            pad = self.padding * dilation
            padded = F.pad(pre, (pad, pad, pad, pad), mode="circular")
            swirl = F.conv2d(padded, kernel.to(self.device), padding=0,
                             groups=3, dilation=dilation)
            swirls.append(swirl)

        # Stage 3: build a spinning phase field from the multi-scale stack
        if not swirls:
            fused = pre
        else:
            base = torch.tanh(swirls[0])
            if len(swirls) > 1:
                phase = torch.atan2(swirls[-1], swirls[0] + 1e-6)
                spiral = base * torch.cos(phase) + torch.sin(phase)
                spiral = spiral + self.spiral_intensity * torch.sin(phase * 2)
                fused = base * 0.6 + torch.tanh(spiral) * 0.4
            else:
                fused = base

            if len(swirls) > 2:
                extra = sum(torch.tanh(s) for s in swirls[1:-1]) / (len(swirls) - 1)
                fused = fused + 0.35 * extra

        # Stage 4: color fuse and residual blend
        fused = F.conv2d(fused, self.kernels[-1].to(self.device), padding=0)
        fused = torch.tanh(fused).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * fused
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        # Keep on device for efficiency
        return result
    
    def get_soul_sliders(self):
        """Return Ophanim-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1
            },
            {
                "label": "Spiral Intensity",
                "value_attr": "spiral_intensity",
                "max_value": 0.5
            }
        ]
