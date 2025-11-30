"""
Leviathan - Abyssal U-Net soul.

Coils multi-scale currents through a bent U-Net spine, letting fine scratches
and coarse swells collide. Shear and ripple fields keep the flow from settling,
so patterns keep knitting into new forms.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Leviathan(Soul):
    """
    Bent U-Net with interference fields and temporal shear.

    Encodes twice, stirs a dilated bottleneck, then rises with skip braids,
    edge scratches, and a ripple shear that steers energy sideways.
    """

    def __init__(self, kernel_size=5, latent_channels=12, drift_magnitude=0.002,
                 momentum=0.8, skip_blend=0.45, ripple_strength=0.28,
                 tide_strength=0.35, depth_mix=0.6, bottleneck_dilation=2,
                 nonlinearity_scale=0.7, device=None):
        """
        Args:
            kernel_size: Spatial kernel size for coarse paths.
            latent_channels: Width of latent streams.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            skip_blend: Weight of skip injections (0-1).
            ripple_strength: Strength of coordinate ripple modulation.
            tide_strength: Strength of shear coupling across axes.
            depth_mix: Blend between coarse and fine detail (0-1).
            bottleneck_dilation: Dilation factor in the bottleneck swirl.
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.skip_blend = skip_blend
        self.ripple_strength = ripple_strength
        self.tide_strength = tide_strength
        self.depth_mix = depth_mix
        self.bottleneck_dilation = bottleneck_dilation
        self.nonlinearity_scale = nonlinearity_scale
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create braid, encoders, bottleneck, decoders, gate, edge, and readout."""
        k = []

        # Color braid
        pre = torch.randn(self.latent_channels, 3, 1, 1)
        pre = pre - pre.mean(dim=(1, 2, 3), keepdim=True)
        pre = pre / (pre.std() + 1e-6)
        k.append(pre)

        # Down path
        enc1 = torch.randn(self.latent_channels, self.latent_channels,
                           self.kernel_size, self.kernel_size)
        enc1 = enc1 - enc1.mean(dim=(2, 3), keepdim=True)
        enc1 = enc1 / (enc1.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(enc1)

        enc2 = torch.randn(self.latent_channels, self.latent_channels,
                           self.kernel_size, self.kernel_size)
        enc2 = enc2 - enc2.mean(dim=(2, 3), keepdim=True)
        enc2 = enc2 / (enc2.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(enc2)

        # Bottleneck swirl (dilated)
        bottleneck = torch.randn(self.latent_channels, self.latent_channels,
                                 self.kernel_size, self.kernel_size)
        bottleneck = bottleneck - bottleneck.mean(dim=(2, 3), keepdim=True)
        bottleneck = bottleneck / (bottleneck.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(bottleneck)

        # Up path
        dec1 = torch.randn(self.latent_channels, self.latent_channels,
                           self.kernel_size, self.kernel_size)
        dec1 = dec1 - dec1.mean(dim=(2, 3), keepdim=True)
        dec1 = dec1 / (dec1.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(dec1)

        dec2 = torch.randn(self.latent_channels, self.latent_channels,
                           self.kernel_size, self.kernel_size)
        dec2 = dec2 - dec2.mean(dim=(2, 3), keepdim=True)
        dec2 = dec2 / (dec2.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(dec2)

        # Gate to modulate energy
        gate = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        gate = gate - gate.mean(dim=(0, 2, 3), keepdim=True)
        gate = gate / (gate.std() + 1e-6)
        k.append(gate)

        # Edge scratch (depthwise)
        edge = torch.randn(self.latent_channels, 1, 3, 3)
        edge = edge - edge.mean(dim=(2, 3), keepdim=True)
        edge = edge / (edge.std() * 9 + 1e-6)
        k.append(edge)

        # Readout to RGB
        readout = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        readout = readout - readout.mean(dim=(2, 3), keepdim=True)
        readout = readout / (readout.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(readout)

        return k

    def apply(self, image, residual_alpha=0.2):
        """
        Apply bent U-Net processing with ripple and shear fields.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new frame versus the old.

        Returns:
            Processed image tensor (C, H, W) on device.
        """
        pad = self.padding
        img = image.to(self.device).unsqueeze(0)

        # Stage 1: braid
        pre = torch.tanh(F.conv2d(img, self.kernels[0].to(self.device), padding=0))

        # Stage 2: down path
        d1 = F.conv2d(F.pad(pre, (pad, pad, pad, pad), mode="circular"),
                      self.kernels[1].to(self.device), stride=2, padding=0)
        d1 = torch.tanh(d1)
        bottleneck_dilation = int(self.bottleneck_dilation)
        d2_pad = pad * bottleneck_dilation
        d2 = F.conv2d(F.pad(d1, (d2_pad, d2_pad, d2_pad, d2_pad), mode="circular"),
                      self.kernels[2].to(self.device), stride=2, padding=0,
                      dilation=bottleneck_dilation)
        d2 = torch.tanh(d2)

        # Stage 3: bottleneck swirl
        bottleneck_pad = pad * bottleneck_dilation
        bottleneck = F.conv2d(F.pad(d2, (bottleneck_pad, bottleneck_pad, bottleneck_pad, bottleneck_pad),
                                    mode="circular"),
                              self.kernels[3].to(self.device), padding=0,
                              dilation=bottleneck_dilation)
        bottleneck = torch.tanh(bottleneck)

        # Stage 4: up path with skip braids
        u1 = F.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=False)
        u1 = F.conv2d(F.pad(u1, (pad, pad, pad, pad), mode="circular"),
                      self.kernels[4].to(self.device), padding=0)
        u1 = torch.tanh(u1)
        if u1.shape[-2:] != d1.shape[-2:]:
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = (1 - self.skip_blend) * u1 + self.skip_blend * d1

        u2 = F.interpolate(u1, scale_factor=2, mode="bilinear", align_corners=False)
        u2 = F.conv2d(F.pad(u2, (pad, pad, pad, pad), mode="circular"),
                      self.kernels[5].to(self.device), padding=0)
        u2 = torch.tanh(u2)
        if u2.shape[-2:] != pre.shape[-2:]:
            u2 = F.interpolate(u2, size=pre.shape[-2:], mode="bilinear", align_corners=False)
        u2 = (1 - self.skip_blend) * u2 + self.skip_blend * pre

        # Stage 5: gate, edge scratches, and ripple interference
        gate = torch.sigmoid(F.conv2d(u2, self.kernels[6].to(self.device), padding=0))

        edge = F.conv2d(F.pad(pre, (1, 1, 1, 1), mode="circular"),
                        self.kernels[7].to(self.device), padding=0,
                        groups=self.latent_channels)
        edge = torch.tanh(edge)

        # Coordinate ripple field
        _, _, h, w = u2.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        phase = torch.mean(bottleneck) * 3.5 + torch.mean(d2)
        ripple = torch.sin((x * 3.1 + y * 4.2) * self.depth_mix + phase)
        ripple = ripple.unsqueeze(0).unsqueeze(0)

        swirl = (1 - self.depth_mix) * u2 + self.depth_mix * edge
        swirl = swirl * (1 + self.ripple_strength * ripple)
        swirl = swirl * (0.6 + 0.4 * gate)

        # Stage 6: shear coupling to steer flow
        shear = torch.roll(swirl, shifts=1, dims=2) - torch.roll(swirl, shifts=-1, dims=3)
        fused = swirl + self.tide_strength * shear

        # Stage 7: readout and residual
        fused = F.conv2d(F.pad(fused, (pad, pad, pad, pad), mode="circular"),
                         self.kernels[8].to(self.device), padding=0)
        fused = torch.tanh(fused).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * fused
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)

        return result

    def get_soul_sliders(self):
        """Return Leviathan-specific sliders."""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Skip Blend",
                "value_attr": "skip_blend",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Ripple Strength",
                "value_attr": "ripple_strength",
                "min_value": 0.0,
                "max_value": 0.6
            },
            {
                "label": "Tide Strength",
                "value_attr": "tide_strength",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Depth Mix",
                "value_attr": "depth_mix",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Bottleneck Dilation",
                "value_attr": "bottleneck_dilation",
                "min_value": 1,
                "max_value": 4
            }
        ]
