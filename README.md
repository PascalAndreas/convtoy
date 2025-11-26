# Convolution Art Toy

An interactive real-time visualization tool that uses convolution layers and ECG signals to create dynamic, organic patterns.

## Features

- **Multi-layer convolution processing** with residual connections
- **ECG-driven kernel drift** - The convolution kernels evolve based on a simulated heartbeat
- **Interactive perturbations** - Click and hold to locally perturb the image
- **Fullscreen mode** - Press F to toggle fullscreen visualization
- **Customizable parameters** - Adjust drift scale, perturbation, and residual mixing in real-time

## Architecture

### Core Components

1. **`convolutions.py`** - Convolution processor module
   - `ConvolutionProcessor` - Abstract base class
   - `MultiLayerConvolution` - Multi-layer implementation with residual connections
   - `SingleLayerConvolution` - Simpler single-layer variant
   
2. **`heart.py`** - ECG signal simulator
   - `ECGSimulator` - Generates realistic ECG waveforms with P, QRS, and T waves
   - Configurable BPM and sample rate
   - Based on Gaussian functions modeling cardiac electrical activity

3. **`main.py`** - Main application
   - `ConvolutionArt` - Pygame-based UI and rendering
   - Integrates convolution processors with ECG-driven drift
   - Handles user interaction and visualization

## Usage

### Basic Usage

```python
# Default configuration (60 BPM, 5-layer convolution)
python main.py
```

### Custom Configuration

```python
from convolutions import MultiLayerConvolution, SingleLayerConvolution
from main import ConvolutionArt

# Faster heart rate
app = ConvolutionArt(bpm=80)

# Single-layer convolution with large kernels
app = ConvolutionArt(
    conv_processor=SingleLayerConvolution(kernel_size=9),
    bpm=45
)

# Custom multi-layer setup
app = ConvolutionArt(
    conv_processor=MultiLayerConvolution(kernel_size=5, num_layers=3),
    bpm=120
)

app.run()
```

## Controls

### Keyboard
- **I** - Randomize image
- **K** - Randomize kernels
- **F** - Toggle fullscreen
- **ESC** - Quit

### Mouse
- **Click & Hold** - Create localized perturbations in the image
- **Sliders** - Adjust parameters:
  - **Drift Scale (ECG)** - How much the ECG signal affects kernel drift
  - **Perturbation** - Amount of noise added to center and edges
  - **Residual Mix** - Blending factor for convolution updates

## ECG Signal

The ECG simulator generates a realistic cardiac waveform consisting of:

- **P wave** - Atrial depolarization (small bump before main spike)
- **QRS complex** - Ventricular depolarization (main spike with Q and S dips)
- **T wave** - Ventricular repolarization (rounded wave after spike)

The signal is constructed using Gaussian functions and automatically loops through heartbeat cycles. The drift applied to convolution kernels is modulated by the absolute value of the ECG signal, creating organic, pulsing visual dynamics.

### Testing the ECG Signal

To visualize the ECG waveform:

```bash
python test_ecg.py
```

This will generate a plot showing 3 seconds of ECG signal.

## Extending the System

### Creating New Convolution Processors

Extend the `ConvolutionProcessor` base class:

```python
from convolutions import ConvolutionProcessor
import torch

class MyCustomProcessor(ConvolutionProcessor):
    def __init__(self, kernel_size=7, device=None):
        self.kernel_size = kernel_size
        super().__init__(padding=kernel_size // 2, device=device)
    
    def _initialize_kernels(self):
        # Your custom kernel initialization
        return [torch.randn(3, 3, self.kernel_size, self.kernel_size)]
    
    def apply(self, image, residual_alpha=0.2):
        # Your custom convolution logic
        return processed_image
```

### Creating New Signal Generators

The drift can be driven by any signal generator that implements a `beat()` method:

```python
class CustomSignalGenerator:
    def __init__(self, sample_rate=60):
        self.sample_rate = sample_rate
        self.time = 0.0
    
    def beat(self):
        # Generate and return next signal value
        signal = your_signal_function(self.time)
        self.time += 1.0 / self.sample_rate
        return signal
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- Pygame >= 2.5.0
- NumPy >= 1.24.0

## Installation

```bash
pip install -r requirements.txt
```

## Technical Details

### Convolution Dynamics

The multi-layer convolution applies sequential transformations:

1. **Circular padding** - Ensures toroidal topology (edges wrap)
2. **3-channel convolution** - Allows color channel mixing
3. **Tanh nonlinearity** - Prevents explosion while maintaining dynamics
4. **Residual connections** - Prevents washout by blending old and new states
5. **Standardization** - Maintains stable contrast structure

### Kernel Design

Kernels are initialized with:
- Zero mean (DC mode killed) - Prevents color collapse
- Controlled spectral radius - Prevents explosion
- Random spatial structure - Creates complex patterns

### ECG-Driven Drift

The drift mechanism:
1. ECG signal samples at frame rate (60 Hz)
2. Absolute value taken to ensure positive drift
3. Scaled by user-controlled `drift_scale` parameter
4. Applied as small random perturbations to kernel weights
5. Kernels re-normalized to maintain stability

This creates organic, pulsing evolution synchronized with the simulated heartbeat.

## License

MIT License - Feel free to use and modify!

