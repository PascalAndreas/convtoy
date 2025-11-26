"""
Convolution processors for image manipulation.

This module provides a base class and concrete implementations for different
convolution-based image processing strategies.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod


class Soul(ABC):
    """
    Base class for convolution-based image processors.
    
    Subclasses should implement the convolution logic and can customize
    kernel initialization, drift behavior, and layer composition.
    """
    
    def __init__(self, padding=3, drift_magnitude=0.002, momentum=0.9, device=None):
        """
        Initialize the convolution processor.
        
        Args:
            padding: Padding size for image (kernel_size // 2, default: 3)
            drift_magnitude: Magnitude of drift direction vector
            momentum: Momentum factor for drift direction updates (0-1)
            device: PyTorch device (cuda or cpu)
        """
        self.padding = padding
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.drift_magnitude = drift_magnitude
        self.momentum = momentum
        
        # Initialize kernels
        self.kernels = self._initialize_kernels()
        
        # Initialize drift directions (one per kernel, on unit sphere)
        self.drift_directions = []
        self.drift_velocities = []  # For momentum
        for kernel in self.kernels:
            # Initialize random direction on unit sphere
            direction = torch.randn_like(kernel)
            direction = direction / (direction.norm() + 1e-8)
            self.drift_directions.append(direction)
            
            # Initialize velocity to zero
            velocity = torch.zeros_like(kernel)
            self.drift_velocities.append(velocity)
    
    @abstractmethod
    def _initialize_kernels(self):
        """
        Initialize convolution kernels.
        
        Returns:
            List of kernel tensors
        """
        pass
    
    @abstractmethod
    def apply(self, image, residual_alpha=0.2):
        """
        Apply convolution processing to an image.
        
        Args:
            image: Input image tensor (C, H, W)
            residual_alpha: Blending factor for residual connections
            
        Returns:
            Processed image tensor (C, H, W)
        """
        pass
    
    def randomize_kernels(self):
        """Randomize all kernels and reset drift directions."""
        self.kernels = self._initialize_kernels()
        
        # Reset drift directions and velocities
        self.drift_directions = []
        self.drift_velocities = []
        for kernel in self.kernels:
            direction = torch.randn_like(kernel)
            direction = direction / (direction.norm() + 1e-8)
            self.drift_directions.append(direction)
            
            velocity = torch.zeros_like(kernel)
            self.drift_velocities.append(velocity)
    
    def change_drift(self, change_amount):
        """
        Update drift directions via random walk on unit sphere with momentum.
        
        This method updates the direction of drift for each kernel. The drift
        direction performs a random walk on an n-dimensional unit sphere, where
        n is the number of elements in each kernel.
        
        Args:
            change_amount: Amount of change to apply to drift direction
        """
        if change_amount > 0:
            for i in range(len(self.kernels)):
                # Generate random tangent perturbation
                tangent_noise = torch.randn_like(self.drift_directions[i]) * change_amount
                
                # Apply momentum to the velocity
                self.drift_velocities[i] = (self.momentum * self.drift_velocities[i] + 
                                           (1 - self.momentum) * tangent_noise)
                
                # Update direction with velocity
                new_direction = self.drift_directions[i] + self.drift_velocities[i]
                
                # Project back onto unit sphere
                new_direction = new_direction / (new_direction.norm() + 1e-8)
                
                self.drift_directions[i] = new_direction
    
    def apply_drift(self, heart_signal):
        """
        Apply drift to kernels using the heart signal and current drift directions.
        
        The drift is applied in the direction of the current drift direction vector,
        scaled by the drift magnitude and the heart signal.
        
        Args:
            heart_signal: Signal value from the heart (typically from ECG)
        """
        if abs(heart_signal) > 1e-8:
            for i in range(len(self.kernels)):
                # Apply drift in the current direction
                drift = self.drift_directions[i] * self.drift_magnitude * heart_signal
                self.kernels[i] = self.kernels[i] + drift
                
                # Re-normalize to prevent explosion
                self.kernels[i] = self.kernels[i] / (self.kernels[i].abs().sum() + 1e-6)
