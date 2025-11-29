"""
Powers - Self-modulating multi-path convolution soul.

Pure convolution architecture where each layer's output generates spatial
attention that modulates subsequent layers. No tricks, no CV gimmicks - just
kernels, feedback, and emergent dynamics. Like Seraphim but self-aware.
"""

import torch
import torch.nn.functional as F

from .base import Soul


class Powers(Soul):
    """
    Multi-layer processor with self-modulating spatial attention.

    Each convolution layer generates both features and attention maps.
    The attention guides how subsequent layers process the image, creating
    feedback loops and emergent behavior from pure convolution dynamics.
    """

    def __init__(self, kernel_size=5, num_layers=6, drift_magnitude=0.015,
                 momentum=0.7, attention_strength=0.5, feature_mix=0.6,
                 layer_skip=0.3, channel_interaction=0.4, compression=0.8,
                 nonlinearity_scale=0.7, device=None):
        """
        Args:
            kernel_size: Size of convolution kernels.
            num_layers: Number of processing layers (default: 6).
            drift_magnitude: Magnitude of kernel drift per heart tick.
            momentum: Momentum for drift direction updates.
            attention_strength: Strength of self-modulation (default: 0.5).
            feature_mix: How much to blend layer outputs (default: 0.6).
            layer_skip: Strength of skip connections (default: 0.3).
            channel_interaction: Cross-channel mixing strength (default: 0.4).
            compression: Nonlinearity compression before attention (default: 0.8).
            nonlinearity_scale: Scale factor for final tanh (default: 0.7).
            device: torch device.
        """
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.attention_strength = attention_strength
        self.feature_mix = feature_mix
        self.layer_skip = layer_skip
        self.channel_interaction = channel_interaction
        self.compression = compression
        self.nonlinearity_scale = nonlinearity_scale
        
        super().__init__(padding=kernel_size // 2, drift_magnitude=drift_magnitude,
                         momentum=momentum, device=device)

    def _initialize_kernels(self):
        """
        Create kernels for each layer.
        
        Each layer gets a full 3x3 channel convolution kernel.
        The variety comes from random initialization and drift.
        """
        kernels = []
        
        for i in range(self.num_layers):
            # Standard 3-in, 3-out convolution
            kernel = torch.randn(3, 3, self.kernel_size, self.kernel_size)
            
            # Zero mean
            kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
            
            # Unit variance
            kernel = kernel / (kernel.std() * (self.kernel_size ** 2) + 1e-6)
            
            # Slight layer-dependent bias toward different characteristics
            layer_factor = i / max(1, self.num_layers - 1)
            
            # Early layers: slightly smoother
            # Late layers: slightly sharper
            if layer_factor < 0.5:
                # Smooth early layers a bit
                kernel = kernel * (0.8 + 0.4 * layer_factor)
            
            kernels.append(kernel)
        
        return kernels

    def _generate_attention(self, features, strength):
        """
        Generate spatial attention map from features.
        
        Args:
            features: (1, C, H, W) feature tensor
            strength: attention strength scalar
            
        Returns:
            attention: (1, 1, H, W) spatial attention map
        """
        # Compute feature energy per spatial location
        energy = features.abs().mean(dim=1, keepdim=True)
        
        # Compress with tanh
        energy = torch.tanh(energy * self.compression)
        
        # Light spatial smoothing for coherence
        if features.shape[2] > 3 and features.shape[3] > 3:
            energy_smooth = F.avg_pool2d(
                F.pad(energy, (1, 1, 1, 1), mode='replicate'),
                kernel_size=3, stride=1
            )
            energy = 0.7 * energy + 0.3 * energy_smooth
        
        # Convert to attention weights centered at 1
        attention = 1.0 + strength * (energy - energy.mean())
        
        return attention

    def apply(self, image, residual_alpha=0.25):
        """
        Apply self-modulating multi-layer convolution.

        Args:
            image: Input image tensor (C, H, W) on any device.
            residual_alpha: Blend factor for the new transformation.
            
        Returns:
            Processed image tensor (C, H, W) on device
        """
        img = image.to(self.device)
        _, height, width = img.shape
        
        # Start with input
        current = img.unsqueeze(0)
        accumulated = torch.zeros_like(current)
        
        # For skip connections
        layer_outputs = []
        
        pad_size = self.padding
        
        for layer_idx, kernel in enumerate(self.kernels):
            # Apply circular padding
            padded = F.pad(current, (pad_size, pad_size, pad_size, pad_size), mode='circular')
            
            # Convolution
            conv_out = F.conv2d(padded, kernel.to(self.device), padding=0)
            
            # Generate attention from current features (before nonlinearity)
            if layer_idx > 0:
                attention = self._generate_attention(conv_out, self.attention_strength)
                # Modulate the convolution output
                conv_out = conv_out * attention
            
            # Nonlinearity
            conv_out = torch.tanh(conv_out)
            
            # Channel interaction: light 1x1 mixing using first kernel's color mixing
            if self.channel_interaction > 0 and layer_idx > 0:
                # Use part of the first kernel as a 1x1 color mixer
                color_mix = self.kernels[0][:, :, self.kernel_size//2, self.kernel_size//2].unsqueeze(-1).unsqueeze(-1)
                color_mix = color_mix.to(self.device)
                mixed = F.conv2d(conv_out, color_mix, padding=0)
                conv_out = (1 - self.channel_interaction) * conv_out + self.channel_interaction * mixed
            
            # Skip connection from earlier layer
            if layer_idx > 1 and self.layer_skip > 0:
                skip_source = layer_outputs[layer_idx // 2]
                if skip_source.shape == conv_out.shape:
                    conv_out = conv_out + self.layer_skip * skip_source
            
            # Store for skip connections
            layer_outputs.append(conv_out)
            
            # Accumulate features
            accumulated = accumulated + self.feature_mix * conv_out
            
            # Update current for next layer (with residual from input)
            current = (1 - residual_alpha * 0.5) * current + (residual_alpha * 0.5) * conv_out
        
        # Final blend: accumulated features + last layer + original
        result = 0.4 * accumulated / len(self.kernels) + 0.4 * current + 0.2 * img.unsqueeze(0)
        
        # Normalize
        result = result.squeeze(0)
        result = result - result.mean(dim=(1, 2), keepdim=True)
        result = result / (result.std(dim=(1, 2), keepdim=True) + 1e-6)
        result = torch.tanh(result * self.nonlinearity_scale)
        
        return result
    
    def get_soul_sliders(self):
        """Return Powers-specific sliders"""
        return [
            {
                "label": "Nonlinearity",
                "value_attr": "nonlinearity_scale",
                "min_value": 0.1,
                "max_value": 1.0
            },
            {
                "label": "Attention Strength",
                "value_attr": "attention_strength",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Feature Mix",
                "value_attr": "feature_mix",
                "min_value": 0.0,
                "max_value": 1.0
            },
            {
                "label": "Layer Skip",
                "value_attr": "layer_skip",
                "min_value": 0.0,
                "max_value": 0.6
            },
            {
                "label": "Channel Interaction",
                "value_attr": "channel_interaction",
                "min_value": 0.0,
                "max_value": 0.8
            },
            {
                "label": "Compression",
                "value_attr": "compression",
                "min_value": 0.3,
                "max_value": 1.5
            }
        ]

