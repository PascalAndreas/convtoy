"""
Metatron - Interleaved resonance soul.

Braids coarse and fine fields into a spinning lattice, then leans on a
slow memory trace to let structures accrete instead of just flicker.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Metatron(Soul):
    """
    Resonant tri-stream processor that encourages emergent interference.

    Uses a color braid, coarse swirl, fine scratch, and a temporal memory
    to keep motion evolving. Cross-coupling and shear encourage patterns to
    collide rather than settle into a single trick.
    """

    def __init__(self, kernel_size=5, latent_channels=8, drift_magnitude=0.0025,
                 momentum=0.82, cross_coupling=0.35, phase_twist=0.45,
                 memory_decay=0.18, shear_strength=0.3, resonance=0.55,
                 nonlinearity_scale=0.65, device=None):
        """
        Args:
            kernel_size: Spatial kernel size for coarse/readout paths.
            latent_channels: Width of the latent resonance field.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            cross_coupling: How much phase interference feeds back (0-1).
            phase_twist: Amount of angular twist applied to phase lattice.
            memory_decay: Strength of temporal memory injection.
            shear_strength: Strength of shear roll coupling across axes.
            resonance: Blend between coarse swirl and fine scratch (0-1).
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.cross_coupling = cross_coupling
        self.phase_twist = phase_twist
        self.memory_decay = memory_decay
        self.shear_strength = shear_strength
        self.resonance = resonance
        self.nonlinearity_scale = nonlinearity_scale
        self.prev_latent = None  # Temporal trace for emergent motifs
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create braid, coarse, fine, gate, and readout kernels."""
        kernels = []

        # Color braid into latent
        braid = torch.randn(self.latent_channels, 3, 1, 1)
        braid = braid - braid.mean(dim=(1, 2, 3), keepdim=True)
        braid = braid / (braid.std() + 1e-6)
        kernels.append(braid)

        # Coarse swirl (learned orientation field)
        coarse = torch.randn(self.latent_channels, self.latent_channels,
                             self.kernel_size, self.kernel_size)
        coarse = coarse - coarse.mean(dim=(2, 3), keepdim=True)
        coarse = coarse / (coarse.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(coarse)

        # Fine scratch (depthwise, higher dilation)
        fine = torch.randn(self.latent_channels, 1, 3, 3)
        fine = fine - fine.mean(dim=(2, 3), keepdim=True)
        fine = fine / (fine.std() * 9 + 1e-6)
        kernels.append(fine)

        # Gate/coupler to modulate interference energy
        gate = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        gate = gate - gate.mean(dim=(0, 2, 3), keepdim=True)
        gate = gate / (gate.std() + 1e-6)
        kernels.append(gate)

        # Readout back to RGB with spatial personality
        readout = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        readout = readout - readout.mean(dim=(2, 3), keepdim=True)
        readout = readout / (readout.std() * (self.kernel_size ** 2) + 1e-6)
        kernels.append(readout)

        return kernels

    def apply(self, image, residual_alpha=0.22):
        """
        Apply resonant tri-stream processing with temporal memory.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new frame versus the old.

        Returns:
            Processed image tensor (C, H, W) on device.
        """
        pad = self.padding
        x = image.to(self.device).unsqueeze(0)

        # Stage 1: color braid
        latent = F.conv2d(x, self.kernels[0].to(self.device), padding=0)
        latent = torch.tanh(latent)

        # Stage 2: coarse swirl (broad strokes)
        coarse = F.pad(latent, (pad, pad, pad, pad), mode="circular")
        coarse = F.conv2d(coarse, self.kernels[1].to(self.device), padding=0)
        coarse = torch.tanh(coarse)

        # Stage 3: fine scratch (local turbulence)
        fine = F.pad(latent, (2, 2, 2, 2), mode="circular")
        fine = F.conv2d(fine, self.kernels[2].to(self.device), padding=0,
                        groups=self.latent_channels, dilation=2)
        fine = torch.tanh(fine)

        # Stage 4: phase interference and gate-controlled coupling
        phase = torch.atan2(fine, coarse + 1e-6)
        phase = torch.tanh(phase * self.phase_twist)
        interference = torch.sin(phase) * torch.cos(torch.roll(phase, shifts=1, dims=3))
        gate = torch.sigmoid(F.conv2d(latent + 0.4 * coarse, self.kernels[3].to(self.device), padding=0))

        spin = (1 - self.resonance) * coarse + self.resonance * fine
        spin = spin + self.cross_coupling * interference
        spin = spin * (0.6 + 0.4 * gate)

        # Stage 5: shear coupling across axes to avoid static symmetry
        shear = torch.roll(spin, shifts=1, dims=2) - torch.roll(spin, shifts=-1, dims=3)
        fused = spin + self.shear_strength * shear

        # Stage 6: temporal memory keeps motifs from dying out
        if self.prev_latent is None:
            memory = torch.zeros_like(fused)
        else:
            memory = self.prev_latent.to(self.device)
        fused = fused + self.memory_decay * memory
        self.prev_latent = fused.detach().cpu()

        # Stage 7: readout and residual blend
        fused = F.pad(fused, (pad, pad, pad, pad), mode="circular")
        out = F.conv2d(fused, self.kernels[4].to(self.device), padding=0)
        out = torch.tanh(out).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * out
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)

        return result

    def get_soul_sliders(self):
        """Return Metatron-specific sliders."""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Cross Coupling",
                "value_attr": "cross_coupling",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Phase Twist",
                "value_attr": "phase_twist",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Memory Decay",
                "value_attr": "memory_decay",
                "min_value": 0.0,
                "max_value": 0.6
            },
            {
                "label": "Shear Strength",
                "value_attr": "shear_strength",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Resonance",
                "value_attr": "resonance",
                "min_value": 0.0,
                "max_value": 1.0
            }
        ]
