"""
Abaddon - Chaos engine soul.

Thrashes a bent U-like spine with coordinate warps, glitch masks, and
interfering depths. Nothing stays still; patterns splinter, rebind, and
fracture into fresh noise.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Abaddon(Soul):
    """
    Chaos-forward processor with warped skips and glitch masks.

    Mixes color, smears through depthwise swirls, warps coordinates with
    evolving phase noise, then injects stochastic scratches before readout.
    """

    def __init__(self, kernel_size=5, latent_channels=14, drift_magnitude=0.0028,
                 momentum=0.78, warp_strength=0.35, warp_freq=3.3,
                 glitch_strength=0.22, glitch_density=0.35, chaos_rate=0.4,
                 depth_mix=0.55, nonlinearity_scale=0.75, device=None):
        """
        Args:
            kernel_size: Spatial kernel size for readout/large depthwise.
            latent_channels: Width of the latent chaos field.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            warp_strength: Amplitude of coordinate warps.
            warp_freq: Frequency of warp field oscillations.
            glitch_strength: Magnitude of injected glitch noise.
            glitch_density: Probability of glitch injection per element.
            chaos_rate: Phase evolution speed for warp field.
            depth_mix: Blend between small and large depthwise swirls.
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.warp_strength = warp_strength
        self.warp_freq = warp_freq
        self.glitch_strength = glitch_strength
        self.glitch_density = glitch_density
        self.chaos_rate = chaos_rate
        self.depth_mix = depth_mix
        self.nonlinearity_scale = nonlinearity_scale
        self.phase = torch.zeros(3)  # Phase state for warp field
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create mix, depthwise swirls, scramble, gate, and readout kernels."""
        kernels = []

        # Color mix
        mix = torch.randn(self.latent_channels, 3, 1, 1)
        mix = mix - mix.mean(dim=(1, 2, 3), keepdim=True)
        mix = mix / (mix.std() + 1e-6)
        kernels.append(mix)

        # Small depthwise scratch
        small = torch.randn(self.latent_channels, 1, 3, 3)
        small = small - small.mean(dim=(2, 3), keepdim=True)
        small = small / (small.std() * 9 + 1e-6)
        kernels.append(small)

        # Large depthwise swirl (dilated)
        large = torch.randn(self.latent_channels, 1, self.kernel_size, self.kernel_size)
        large = large - large.mean(dim=(2, 3), keepdim=True)
        large = large / (large.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(large)

        # Scramble mixer (1x1) for chaotic channel blending
        scramble = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        scramble = scramble - scramble.mean(dim=(0, 2, 3), keepdim=True)
        scramble = scramble / (scramble.std() + 1e-6)
        kernels.append(scramble)

        # Gate to modulate energy
        gate = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        gate = gate - gate.mean(dim=(0, 2, 3), keepdim=True)
        gate = gate / (gate.std() + 1e-6)
        kernels.append(gate)

        # Readout back to RGB
        readout = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        readout = readout - readout.mean(dim=(2, 3), keepdim=True)
        readout = readout / (readout.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(readout)

        return kernels

    def apply(self, image, residual_alpha=0.18):
        """
        Apply chaotic warped convolutional processing.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new chaos versus the old.

        Returns:
            Processed image tensor (C, H, W) on device.
        """
        pad = self.padding
        img = image.to(self.device).unsqueeze(0)

        # Stage 1: color mix
        latent = torch.tanh(F.conv2d(img, self.kernels[0].to(self.device), padding=0))

        # Stage 2: depthwise scratches
        small_pad = 1
        small = F.conv2d(F.pad(latent, (small_pad, small_pad, small_pad, small_pad), mode="circular"),
                         self.kernels[1].to(self.device), padding=0, groups=self.latent_channels)
        small = torch.tanh(small)

        large_pad = pad * 2
        large = F.conv2d(F.pad(latent, (large_pad, large_pad, large_pad, large_pad), mode="circular"),
                         self.kernels[2].to(self.device), padding=0, groups=self.latent_channels, dilation=2)
        large = torch.tanh(large)

        # Stage 3: warp field based on evolving phase and depth energy
        phase = self.phase.to(self.device)
        phase_noise = torch.randn_like(phase)
        phase = phase + self.chaos_rate * phase_noise
        self.phase = phase.detach().cpu()

        _, _, h, w = latent.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        mag = torch.tanh(self.depth_mix * large.mean(dim=1, keepdim=True) +
                         (1 - self.depth_mix) * small.mean(dim=1, keepdim=True))

        warp_x = self.warp_strength * (torch.sin(self.warp_freq * y + phase[0]) +
                                       torch.cos(self.warp_freq * x + phase[1]) +
                                       0.6 * mag.squeeze(1))
        warp_y = self.warp_strength * (torch.cos(self.warp_freq * y - phase[1]) +
                                       torch.sin(self.warp_freq * x + phase[2]) +
                                       0.6 * mag.squeeze(1))
        base_grid = torch.stack((x, y), dim=-1)
        warp_grid = torch.stack((x + warp_x, y + warp_y), dim=-1)
        # Clamp to valid grid_sample range to avoid undefined padding modes
        warp_grid = torch.clamp(warp_grid, -1.0, 1.0)

        # Use zero padding for MPS compatibility (border not supported on MPS)
        warped = F.grid_sample(latent, warp_grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)

        # Stage 4: scramble channels and gate
        scramble = F.conv2d(warped, self.kernels[3].to(self.device), padding=0)
        gate = torch.sigmoid(F.conv2d(warped + small, self.kernels[4].to(self.device), padding=0))
        chaos = scramble + self.depth_mix * large + (1 - self.depth_mix) * small
        chaos = chaos * (0.55 + 0.45 * gate)

        # Stage 5: glitch injection
        if self.glitch_strength > 0 and self.glitch_density > 0:
            mask = (torch.rand_like(chaos) < self.glitch_density).float()
            noise = torch.randn_like(chaos)
            chaos = chaos + self.glitch_strength * noise * mask
            chaos = torch.tanh(chaos)

        # Stage 6: readout and residual
        chaos = F.pad(chaos, (pad, pad, pad, pad), mode="circular")
        out = F.conv2d(chaos, self.kernels[5].to(self.device), padding=0)
        out = torch.tanh(out).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * out
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)

        return result

    def get_soul_sliders(self):
        """Return Abaddon-specific sliders."""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Warp Strength",
                "value_attr": "warp_strength",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Warp Frequency",
                "value_attr": "warp_freq",
                "min_value": 0.5,
                "max_value": 6.0
            },
            {
                "label": "Glitch Strength",
                "value_attr": "glitch_strength",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Glitch Density",
                "value_attr": "glitch_density",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Chaos Rate",
                "value_attr": "chaos_rate",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Depth Mix",
                "value_attr": "depth_mix",
                "min_value": 0.0,
                "max_value": 1.0
            }
        ]
