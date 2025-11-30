"""
Azazel - Chaotic flowfield soul.

Spins depthwise swirls through drifting multi-region fields, then warps
everything with a curl-like flow to keep fragments colliding. Runs best
with low residual for painterly chaos.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Azazel(Soul):
    """
    Flow-driven chaos with drifting regions and curl warps.

    Builds soft regions from oscillating wavefields, mixes small and large
    swirls per region, derives a flow from the energy map, and warps the
    features so nothing stays put. Glitch keeps edges alive.
    """

    def __init__(self, kernel_size=5, latent_channels=14, num_regions=4,
                 drift_magnitude=0.003, momentum=0.78, warp_strength=0.32,
                 warp_speed=0.5, region_sharpness=2.6, depth_mix=0.6,
                 turbulence=0.45, glitch_strength=0.2, nonlinearity_scale=0.72,
                 device=None):
        """
        Args:
            kernel_size: Spatial kernel size for large swirl/readout.
            latent_channels: Width of the latent chaos field.
            num_regions: Number of drifting regions.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            warp_strength: Amplitude of flow-driven warp (0-1).
            warp_speed: Speed of region phase drift.
            region_sharpness: Softmax sharpness for region blending.
            depth_mix: Blend between large and small depthwise swirls.
            turbulence: Strength of shear roll after warp.
            glitch_strength: Magnitude of random glitch injection.
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.num_regions = num_regions
        self.warp_strength = warp_strength
        self.warp_speed = warp_speed
        self.region_sharpness = region_sharpness
        self.depth_mix = depth_mix
        self.turbulence = turbulence
        self.glitch_strength = glitch_strength
        self.nonlinearity_scale = nonlinearity_scale
        self.region_phase = torch.zeros(num_regions, 3)
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create mix, small swirl, large swirl, shuffle, gate, readout."""
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

        # Large depthwise swirl (dilated)
        large = torch.randn(self.latent_channels, 1, self.kernel_size, self.kernel_size)
        large = large - large.mean(dim=(2, 3), keepdim=True)
        large = large / (large.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(large)

        # Shuffle (1x1) to remix after warp
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
        """Generate drifting region weights with richer wave superposition."""
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        phase = self.region_phase.to(self.device)
        phase = phase + self.warp_speed * torch.randn_like(phase)
        self.region_phase = phase.detach().cpu()

        logits = []
        for idx in range(self.num_regions):
            p0, p1, p2 = phase[idx]
            freq = 1.3 + 0.4 * idx
            diag = torch.sin(freq * (x + y) + p0)
            rings = torch.cos(freq * (x**2 + y**2) * 1.8 + p1)
            stripes = torch.sin(freq * (x - y) * 0.8 + p2)
            field = diag + 0.7 * rings + 0.9 * stripes
            logits.append(field)
        logits = torch.stack(logits, dim=0)  # (K, H, W)
        weights = F.softmax(self.region_sharpness * logits, dim=0)
        return weights

    def _flow_grid(self, energy, warp_strength):
        """Derive a flow grid from scalar energy via pseudo-curl."""
        # energy: (1, 1, H, W)
        dy = energy - torch.roll(energy, shifts=1, dims=2)
        dx = torch.roll(energy, shifts=-1, dims=3) - energy
        # Normalize flow to reasonable scale
        scale = energy.abs().mean() + 1e-4
        flow_y = dy.squeeze(1) / scale
        flow_x = dx.squeeze(1) / scale
        return flow_x, flow_y

    def apply(self, image, residual_alpha=0.16):
        """
        Apply flowfield chaos.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new frame versus the old.

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

        big_pad = pad * 2
        large = F.conv2d(F.pad(latent, (big_pad, big_pad, big_pad, big_pad), mode="circular"),
                         self.kernels[2].to(self.device), padding=0,
                         groups=self.latent_channels, dilation=2)
        large = torch.tanh(large)

        base = self.depth_mix * large + (1 - self.depth_mix) * small

        # Stage 3: region blend
        _, _, h, w = base.shape
        weights = self._region_weights(h, w)  # (K, H, W)
        regioned = (weights.unsqueeze(1) * base).sum(dim=0, keepdim=True)

        # Stage 4: flow from energy + swirl warp
        energy = regioned.norm(dim=1, keepdim=True)
        flow_x, flow_y = self._flow_grid(energy, self.warp_strength)

        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')
        swirl = 0.3 * torch.sin(energy.squeeze(1) * 2.5)

        grid_x = x + self.warp_strength * (flow_x + swirl)
        grid_y = y + self.warp_strength * (flow_y - swirl)
        warp_grid = torch.stack((grid_x, grid_y), dim=-1)
        warp_grid = torch.clamp(warp_grid, -1.0, 1.0)

        warped = F.grid_sample(regioned, warp_grid, mode="bilinear",
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
        """Return Azazel-specific sliders."""
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
            },
            {
                "label": "Glitch Strength",
                "value_attr": "glitch_strength",
                "min_value": 0.0,
                "max_value": 0.8
            }
        ]
