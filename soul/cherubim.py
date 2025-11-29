"""
Cherubim - Dual-stream guardian soul.

Splits motion into a fast edge stream and a slower color-breath stream,
then braids them back together with gated feedback for shimmering guards.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Cherubim(Soul):
    """
    Dual-stream separable convolution with gated feedback.

    Expands into a latent field, sharpens edges depthwise, then folds
    the feedback through a color gate to keep motion alive without exploding.
    """

    def __init__(self, kernel_size=5, latent_channels=6, drift_magnitude=0.0018,
                 momentum=0.85, nonlinearity_scale=0.6, edge_strength=0.3, device=None):
        """
        Args:
            kernel_size: Size of encoder/decoder spatial kernels.
            latent_channels: Width of the latent swirl field.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            nonlinearity_scale: Scale factor for final tanh (default: 0.6)
            edge_strength: Strength of edge stream injection (default: 0.3)
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.nonlinearity_scale = nonlinearity_scale
        self.edge_strength = edge_strength
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create encoder, edge, gate, and decoder kernels."""
        kernels = []

        # Encoder: mix colors into a wider latent braid
        enc = torch.randn(self.latent_channels, 3, self.kernel_size, self.kernel_size)
        enc = enc - enc.mean(dim=(-1, -2), keepdim=True)
        enc = enc / (enc.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(enc)

        # Edge stream: per-channel sharpeners (depthwise)
        edge = torch.randn(self.latent_channels, 1, 3, 3)
        edge = edge - edge.mean(dim=(-1, -2), keepdim=True)
        edge = edge / (edge.std() * 9 + 1e-6)
        kernels.append(edge)

        # Gate: 1x1 latent mixer to modulate feedback energy
        gate = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        gate = gate - gate.mean(dim=(0, 2, 3), keepdim=True)
        gate = gate / (gate.std() + 1e-6)
        kernels.append(gate)

        # Decoder: collapse latent back to RGB with a soft spatial touch
        dec = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        dec = dec - dec.mean(dim=(-1, -2), keepdim=True)
        dec = dec / (dec.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(dec)

        return kernels

    def apply(self, image, residual_alpha=0.18):
        """
        Apply dual-stream gated convolutional feedback.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new frame versus the old.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        pad = self.padding
        x = F.pad(image.unsqueeze(0).to(self.device),
                  (pad, pad, pad, pad), mode="circular")

        # Encode into latent swirl field
        latent = F.conv2d(x, self.kernels[0].to(self.device), padding=0)
        latent = torch.tanh(latent)

        # Edge stream (depthwise) to inject crisp turbulence
        edge = F.pad(latent, (1, 1, 1, 1), mode="circular")
        edge = F.conv2d(edge, self.kernels[1].to(self.device), padding=0,
                        groups=self.latent_channels)
        edge = torch.tanh(edge)

        # Gate controls how much feedback to keep each step
        gate = torch.sigmoid(F.conv2d(latent, self.kernels[2].to(self.device), padding=0))

        # Blend latent motion, edge spikes, and a slight spatial roll for motion
        rolled = torch.roll(latent, shifts=(1, -1), dims=(2, 3))
        roll_strength = 1.0 - self.edge_strength  # Complementary strength
        braided = latent + self.edge_strength * edge + roll_strength * rolled
        braided = braided * (0.55 + 0.45 * gate)

        # Decode back to RGB
        braided = F.pad(braided, (pad, pad, pad, pad), mode="circular")
        out = F.conv2d(braided, self.kernels[3].to(self.device), padding=0)
        out = torch.tanh(out).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * out
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        # Keep on device for efficiency
        return result
    
    def get_soul_sliders(self):
        """Return Cherubim-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1
            },
            {
                "label": "Edge Strength",
                "value_attr": "edge_strength",
                "min_value": 0.1,
                "max_value": 0.9
            }
        ]
