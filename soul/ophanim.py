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
                 momentum=0.8, nonlinearity_scale=0.6, spiral_intensity=0.25,
                 magnitude_influence=0.5, spiral_blend=0.4, phase_offset=0.0,
                 num_harmonics=2, harmonic_falloff=1.5, phase_momentum=0.3, device=None):
        """
        Args:
            kernel_size: Base kernel size for the depthwise filters.
            dilations: Iterable of dilation factors for the depthwise stack.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            nonlinearity_scale: Scale factor for final tanh (default: 0.6)
            spiral_intensity: Intensity of spiral phase field (default: 0.25)
            magnitude_influence: How much magnitude modulates spiral (default: 0.5)
            spiral_blend: Blend ratio between base and spiral (default: 0.4)
            phase_offset: Phase rotation offset in radians (default: 0.0)
            num_harmonics: Number of harmonic frequencies to add (default: 2)
            harmonic_falloff: Exponential falloff for harmonics (default: 1.5)
            phase_momentum: Temporal momentum for phase evolution (default: 0.3)
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.nonlinearity_scale = nonlinearity_scale
        self.spiral_intensity = spiral_intensity
        self.magnitude_influence = magnitude_influence
        self.spiral_blend = spiral_blend
        self.phase_offset = phase_offset
        self.num_harmonics = num_harmonics
        self.harmonic_falloff = harmonic_falloff
        self.phase_momentum = phase_momentum
        self.prev_phase = None  # For temporal coherence
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
                # Compute magnitude and phase from complex representation
                magnitude = torch.sqrt(swirls[0]**2 + swirls[-1]**2 + 1e-6)
                phase = torch.atan2(swirls[-1], swirls[0] + 1e-6) + self.phase_offset
                
                # Apply temporal momentum for smoother rotation
                if self.prev_phase is not None:
                    prev_phase_device = self.prev_phase.to(self.device)
                    # Handle phase wraparound (-π to π)
                    phase_diff = phase - prev_phase_device
                    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
                    phase = phase + self.phase_momentum * phase_diff
                self.prev_phase = phase.detach().cpu()
                
                # Normalize magnitude for modulation
                mag_norm = magnitude / (magnitude.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
                mag_modulation = 1.0 + self.magnitude_influence * (mag_norm - 1.0)
                
                # Create spiral with magnitude modulation
                spiral = mag_modulation * (base * torch.cos(phase) + torch.sin(phase))
                
                # Add harmonic series for richer patterns
                for n in range(1, int(self.num_harmonics) + 1):
                    weight = self.spiral_intensity / (n ** self.harmonic_falloff)
                    spiral = spiral + weight * torch.sin(phase * (n + 1))
                
                # Configurable blend between base and spiral
                fused = (1 - self.spiral_blend) * base + self.spiral_blend * torch.tanh(spiral)
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
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Spiral Intensity",
                "value_attr": "spiral_intensity",
                "min_value": 0.0,
                "max_value": 0.5
            },
            {
                "label": "Magnitude Influence",
                "value_attr": "magnitude_influence",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Spiral Blend",
                "value_attr": "spiral_blend",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Phase Offset",
                "value_attr": "phase_offset",
                "min_value": -3.14159,
                "max_value": 3.14159
            },
            {
                "label": "Harmonics",
                "value_attr": "num_harmonics",
                "min_value": 1,
                "max_value": 5
            },
            {
                "label": "Harmonic Falloff",
                "value_attr": "harmonic_falloff",
                "min_value": 1.0,
                "max_value": 3.0
            },
            {
                "label": "Phase Momentum",
                "value_attr": "phase_momentum",
                "min_value": 0.0,
                "max_value": 0.8
            }
        ]
