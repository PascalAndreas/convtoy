"""
Janus - Bifacial phantom soul.

Leans into bilateral symmetry and drifting facial anchors to coax
pareidolia. Works best with low residual to let faces ooze in.
"""

import torch
import torch.nn.functional as F
import math

from .base import Soul


class Janus(Soul):
    """
    Symmetry-driven soul with drifting eye/mouth anchors.

    Blends mirrored features, overlays gentle Gaussian anchors for eyes/mouth,
    and sharpens with edge + blob filters to hint at faces without a model.
    """

    def __init__(self, kernel_size=5, latent_channels=10, drift_magnitude=0.0018,
                 momentum=0.82, symmetry_strength=0.55, anchor_focus=0.6,
                 feature_sharpness=1.2, smile_curve=0.35,
                 nonlinearity_scale=0.65, device=None):
        """
        Args:
            kernel_size: Spatial kernel size for blob/readout filters.
            latent_channels: Width of latent facial features.
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            symmetry_strength: Weight of bilateral mirror blend (0-1).
            anchor_focus: How strongly to emphasize anchors (0-1).
            feature_sharpness: Strength of edge/blob sharpening.
            smile_curve: Controls mouth curvature modulation.
            nonlinearity_scale: Scale for final tanh shaping.
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.latent_channels = latent_channels
        self.symmetry_strength = symmetry_strength
        self.anchor_focus = anchor_focus
        self.feature_sharpness = feature_sharpness
        self.smile_curve = smile_curve
        self.nonlinearity_scale = nonlinearity_scale
        # Drift phases for anchors (eyes, mouth), each with (x,y) offsets
        self.anchor_phase = torch.zeros(3, 2)
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """Create mix, edge, blob, fuse, and readout kernels."""
        k = []

        # Color mix
        mix = torch.randn(self.latent_channels, 3, 1, 1)
        mix = mix - mix.mean(dim=(1, 2, 3), keepdim=True)
        mix = mix / (mix.std() + 1e-6)
        k.append(mix)

        # Edge filter (depthwise)
        edge = torch.randn(self.latent_channels, 1, 3, 3)
        edge = edge - edge.mean(dim=(2, 3), keepdim=True)
        edge = edge / (edge.std() * 9 + 1e-6)
        k.append(edge)

        # Blob filter (depthwise, larger receptive field)
        blob = torch.randn(self.latent_channels, 1, self.kernel_size, self.kernel_size)
        blob = blob - blob.mean(dim=(2, 3), keepdim=True)
        blob = blob / (blob.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(blob)

        # Fuse (1x1) to re-mix latent channels
        fuse = torch.randn(self.latent_channels, self.latent_channels, 1, 1)
        fuse = fuse - fuse.mean(dim=(0, 2, 3), keepdim=True)
        fuse = fuse / (fuse.std() + 1e-6)
        k.append(fuse)

        # Readout to RGB with gentle spatial personality
        readout = torch.randn(3, self.latent_channels, self.kernel_size, self.kernel_size)
        readout = readout - readout.mean(dim=(2, 3), keepdim=True)
        readout = readout / (readout.std() * (self.kernel_size ** 2) + 1e-6)
        k.append(readout)

        return k

    def _anchor_mask(self, h, w):
        """Create drifting Gaussian anchors for eyes and mouth."""
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=self.device),
                              torch.linspace(-1, 1, w, device=self.device), indexing='ij')

        phase = self.anchor_phase.to(self.device)
        drift = 0.08 * torch.randn_like(phase)
        phase = phase + drift
        self.anchor_phase = phase.detach().cpu()

        centers = [
            (-0.32 + 0.12 * torch.sin(phase[0, 0]), -0.18 + 0.1 * torch.sin(phase[0, 1])),  # left eye
            (0.32 + 0.12 * torch.sin(phase[1, 0]), -0.18 + 0.1 * torch.sin(phase[1, 1])),   # right eye
            (0.0 + 0.1 * torch.sin(phase[2, 0]), 0.22 + 0.1 * torch.sin(phase[2, 1]))        # mouth
        ]

        sig_eye = 0.22
        sig_mouth = 0.28

        masks = []
        for idx, (cx, cy) in enumerate(centers):
            sigma = sig_eye if idx < 2 else sig_mouth
            mask = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            if idx == 2:
                # Smile modulation
                smile = torch.sin(math.pi * x) * self.smile_curve
                mask = mask * (1.0 + smile)
            masks.append(mask)

        eyes = masks[0] + masks[1]
        mouth = masks[2]
        anchor = eyes * 1.1 + mouth * 1.3
        anchor = torch.clamp(anchor, 0.0, 2.5)
        return anchor

    def apply(self, image, residual_alpha=0.12):
        """
        Apply symmetry-driven face coaxing.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new frame versus the old.

        Returns:
            Processed image tensor (C, H, W) on device.
        """
        pad = self.padding
        img = image.to(self.device).unsqueeze(0)

        # Stage 1: mix
        latent = torch.tanh(F.conv2d(img, self.kernels[0].to(self.device), padding=0))

        # Stage 2: bilateral blend
        mirrored = torch.flip(latent, dims=[3])
        sym = (1 - self.symmetry_strength) * latent + self.symmetry_strength * 0.5 * (latent + mirrored)

        # Stage 3: features
        edge = F.conv2d(F.pad(sym, (1, 1, 1, 1), mode="circular"),
                        self.kernels[1].to(self.device), padding=0,
                        groups=self.latent_channels)
        edge = torch.tanh(self.feature_sharpness * edge)

        blob = F.conv2d(F.pad(sym, (pad, pad, pad, pad), mode="circular"),
                        self.kernels[2].to(self.device), padding=0,
                        groups=self.latent_channels)
        blob = torch.tanh(blob)

        fused_feat = 0.6 * edge + 0.4 * blob

        # Stage 4: anchor gating
        _, _, h, w = fused_feat.shape
        anchor = self._anchor_mask(h, w)
        gate = torch.sigmoid(self.anchor_focus * anchor).unsqueeze(0).unsqueeze(0)
        guided = fused_feat * (0.7 + 0.3 * gate) + sym * (1 - 0.4 * gate)

        # Stage 5: fuse and readout
        guided = F.conv2d(guided, self.kernels[3].to(self.device), padding=0)
        guided = torch.tanh(guided)

        guided = F.conv2d(F.pad(guided, (pad, pad, pad, pad), mode="circular"),
                          self.kernels[4].to(self.device), padding=0)
        guided = torch.tanh(guided).squeeze(0)

        img_on_device = image.to(self.device)
        result = (1 - residual_alpha) * img_on_device + residual_alpha * guided
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)

        return result

    def get_soul_sliders(self):
        """Return Janus-specific sliders."""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Symmetry Strength",
                "value_attr": "symmetry_strength",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Anchor Focus",
                "value_attr": "anchor_focus",
                "min_value": 0.0,
                "max_value": 1.2
            },
            {
                "label": "Feature Sharpness",
                "value_attr": "feature_sharpness",
                "min_value": 0.2,
                "max_value": 2.0
            },
            {
                "label": "Smile Curve",
                "value_attr": "smile_curve",
                "min_value": 0.0,
                "max_value": 1.0
            }
        ]
