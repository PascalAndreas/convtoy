"""
Pandemonium - Dynamic chaos lattice soul.

Splits motion across drifting regions, swirls at two depths, then warps the
field so shards of detail slam into broad currents. Designed to be run with
low residual for painterly, evolving turbulence.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Pandemonium(Soul):
    """
    Region-driven chaos with moving interference bands.

    Builds multiple soft regions from drifting wave patterns, blends shallow
    and deep depthwise swirls per region, warps the field, and injects glitch
    noise to keep things from settling.
    """

    def __init__(self, kernel_size=5, latent_channels=12, num_regions=4,
                 drift_magnitude=0.003, momentum=0.78, warp_strength=0.3,
                 warp_speed=0.45, glitch_strength=0.18, region_sharpness=2.5,
                 depth_mix=0.55, turbulence=0.4, nonlinearity_scale=0.7, device=None):
        """
        Args:
            kernel_size: Spatial kernel size for the large swirl/readout.
            latent_channels: Width of the latent chaos field.
            num_regions: Number of drifting regions (softly blended).
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            warp_strength: Strength of coordinate warps (0-1).
            warp_speed: Speed of region phase drift (controls pattern motion).
            glitch_strength: Magnitude of random glitch injection.
            region_sharpness: Sharpness of region softmax (>1 is crisper).
            depth_mix: Blend between large and small depthwise swirls.
            turbulence: Amount of shear-like roll added after warp.
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.num_regions = num_regions
        self.warp_strength = warp_strength
        self.warp_speed = warp_speed
        self.glitch_strength = glitch_strength
        self.region_sharpness = region_sharpness
        self.depth_mix = depth_mix
        self.turbulence = turbulence
        self.nonlinearity_scale = nonlinearity_scale
        # Region phase state (two-phase per region) drifts each step
        self.region_phase = torch.zeros(num_regions, 2)
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create mix, small swirl, large swirl, shuffle, gate, and readout kernels."""
        k = []

        # Color mix
        mix = torch.randn(self.latent_channels, 3, 1, 1)
        mix = mix - mix.mean(dim=(1, 2, 3), keepdim=True)
        mix = mix / (mix.std() + 1e-6)
        k.append(mix)

        # Small depthwise swirl
        small = torch.randn(self.latent_channels, 1, 3, 3)
        small = small - small.mean(dim=(2, 3), keepdim=True)
        small = small / (small.std() * 9 + 1e-6)
        k.append(small)

        # Large depthwise swirl (more spatial reach)
        large = torch.randn(self.latent_channels, 1, self.kernel_size, self.kernel_size)
        large = large - large.mean(dim=(2, 3), keepdim=True)
        large = large / (large.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(large)

        # Shuffle (1x1) mixes channels after regional blending
        shuffle = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        shuffle = shuffle - shuffle.mean(dim=(0, 2, 3), keepdim=True)
        shuffle = shuffle / (shuffle.std() + 1e-6)
        k.append(shuffle)

        # Gate to modulate energy
        gate = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        gate = gate - gate.mean(dim=(0, 2, 3), keepdim=True)
        gate = gate / (gate.std() + 1e-6)
        k.append(gate)

        # Readout back to RGB
        readout = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        readout = readout - readout.mean(dim=(2, 3), keepdim=True)
        readout = readout / (readout.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(readout)

        return k

    def _region_weights(self, h, w):
        """Generate drifting soft region weights."""
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')

        # Drift phases
        phase = self.region_phase.to(self.device)
        noise = torch.randn_like(phase)
        phase = phase + self.warp_speed * noise
        self.region_phase = phase.detach().cpu()

        logits = []
        for idx in range(self.num_regions):
            p0, p1 = phase[idx]
            freq = 1.4 + 0.5 * idx
            field = (torch.sin(freq * x + p0) +
                     torch.cos(freq * y - p1) +
                     0.7 * torch.sin(freq * (x + y) + 0.5 * (p0 + p1)))
            logits.append(field)
        logits = torch.stack(logits, dim=0)  # (K, H, W)
        weights = F.softmax(self.region_sharpness * logits, dim=0)
        return weights  # (K, H, W)

    def apply(self, image, residual_alpha=0.15):
        """
        Apply dynamic chaos lattice.

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

        # Stage 2: depthwise swirls
        small = F.conv2d(F.pad(latent, (1, 1, 1, 1), mode="circular"),
                         self.kernels[1].to(self.device), padding=0,
                         groups=self.latent_channels)
        small = torch.tanh(small)

        # For dilation=2, effective kernel grows, so pad more to keep size
        big_pad = pad * 2
        large = F.conv2d(F.pad(latent, (big_pad, big_pad, big_pad, big_pad), mode="circular"),
                         self.kernels[2].to(self.device), padding=0,
                         groups=self.latent_channels, dilation=2)
        large = torch.tanh(large)

        base = self.depth_mix * large + (1 - self.depth_mix) * small

        # Stage 3: region blend
        _, _, h, w = base.shape
        weights = self._region_weights(h, w)  # (K, H, W)
        weighted = (weights.unsqueeze(1) * base).sum(dim=0, keepdim=True)

        # Stage 4: warp based on energy and drifting phases
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        mag = torch.tanh(weighted.norm(dim=1, keepdim=True))

        warp_x = self.warp_strength * (torch.sin(self.warp_speed * y + mag.squeeze(1)) +
                                       torch.cos(self.warp_speed * x - mag.squeeze(1)))
        warp_y = self.warp_strength * (torch.cos(self.warp_speed * y - mag.squeeze(1)) +
                                       torch.sin(self.warp_speed * x + mag.squeeze(1)))
        warp_grid = torch.stack((x + warp_x, y + warp_y), dim=-1)
        warp_grid = torch.clamp(warp_grid, -1.0, 1.0)

        warped = F.grid_sample(weighted, warp_grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)

        # Stage 5: turbulence roll + shuffle + gate
        shear = torch.roll(warped, shifts=1, dims=2) - torch.roll(warped, shifts=-1, dims=3)
        turbulent = warped + self.turbulence * shear

        shuffled = F.conv2d(turbulent, self.kernels[3].to(self.device), padding=0)
        gate = torch.sigmoid(F.conv2d(turbulent, self.kernels[4].to(self.device), padding=0))
        chaos = shuffled * (0.55 + 0.45 * gate)

        # Stage 6: glitch injection
        if self.glitch_strength > 0:
            mask = (torch.rand_like(chaos) < 0.5).float()
            noise = torch.randn_like(chaos)
            chaos = chaos + self.glitch_strength * noise * mask
            chaos = torch.tanh(chaos)

        # Stage 7: readout and residual
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
        """Return Pandemonium-specific sliders."""
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
                "label": "Warp Speed",
                "value_attr": "warp_speed",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Glitch Strength",
                "value_attr": "glitch_strength",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Region Sharpness",
                "value_attr": "region_sharpness",
                "min_value": 0.5,
                "max_value": 5.0
            },
            {
                "label": "Depth Mix",
                "value_attr": "depth_mix",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Turbulence",
                "value_attr": "turbulence",
                "min_value": 0.0,
                "max_value": 0.8
            }
        ]
