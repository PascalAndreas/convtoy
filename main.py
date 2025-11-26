import pygame
import torch
import torch.nn.functional as F
import numpy as np
import sys
from soul import Seraphim, Imp
from heart import Heart

# Initialize Pygame
pygame.init()

# Constants
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER = (90, 90, 90)
TEXT_COLOR = (255, 255, 255)

# UI dimensions
UI_PANEL_WIDTH = 250  # Width of right panel with buttons/sliders
IMAGE_MARGIN = 20  # Margin around image
INSTRUCTIONS_HEIGHT = 220  # Height needed for instructions at bottom

class ConvolutionArt:
    def __init__(self, conv_processor=None, bpm=120):
        # Device (use CUDA if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Convolution processor
        if conv_processor is None:
            conv_processor = Seraphim(kernel_size=7, num_layers=5, device=self.device)
        self.conv_processor = conv_processor
        
        # Heart simulator for driving drift
        # Sample rate matches FPS, amplitude will be scaled by drift_scale
        # HRV adds natural beat-to-beat variation (5% variability, 15 breaths/min)
        self.heart = Heart(bpm=bpm, sample_rate=FPS, amplitude=1.0,
                          hrv_amount=0.05, breathing_rate=0.25)
        
        # Drift scaling factor (multiplies ECG signal)
        self.drift_scale = 0.0
        self.max_drift_scale = 1.0
        
        # Image dimensions (double size, no scaling needed)
        # 512 - 2*padding ensures power of 2 after circular padding
        self.img_width = 512 - 2 * self.conv_processor.padding
        self.img_height = 512 - 2 * self.conv_processor.padding
        
        # Calculate window size dynamically based on image size
        self.window_width = self.img_width + UI_PANEL_WIDTH + IMAGE_MARGIN * 3
        self.window_height = max(self.img_height + IMAGE_MARGIN * 2, INSTRUCTIONS_HEIGHT + 200)
        
        # Create screen with calculated dimensions
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Convolution Art Toy")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Fullscreen mode
        self.fullscreen = False
        self.original_size = (self.window_width, self.window_height)
        self.fullscreen_img_size = None  # Will be set when entering fullscreen
        
        # Initialize random image
        self.image = self._random_image()
        
        # Image noise settings (continuous center + vignette perturbation)
        self.noise_amount = 0.0  # How much random noise to add to image each frame
        self.max_noise = 0.2
        
        # Residual/blend parameter (prevents collapse)
        self.residual_alpha = 0.2  # How much of new conv to blend in (0=no change, 1=full replacement)
        self.max_residual = 1.0
        
        # Precompute center + vignette mask
        self._create_perturbation_mask()
        
        # Mouse perturbation settings
        self.mouse_pressed = False
        self.mouse_press_start = 0
        self.mouse_pos = (0, 0)
        self.perturb_radius = 30
        self.display_offset = (20, 20)  # Where image is drawn on screen
        
        # UI elements
        self.font = pygame.font.Font(None, 24)
        self.buttons = self._create_buttons()
        self.slider = self._create_slider()
        
        # Create reusable surface for efficient pixel updates (no scaling)
        self.display_surface = pygame.Surface((self.img_width, self.img_height))
        
    def _random_image(self):
        """Generate a random RGB image"""
        return torch.rand(3, self.img_height, self.img_width)
    
    def _create_perturbation_mask(self):
        """Create a mask for center + vignette perturbation"""
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.img_height, dtype=torch.float32),
            torch.arange(self.img_width, dtype=torch.float32),
            indexing='ij'
        )
        
        # Center coordinates
        center_x = self.img_width / 2.0
        center_y = self.img_height / 2.0
        
        # Distance from center (normalized)
        dx = (x_coords - center_x) / center_x
        dy = (y_coords - center_y) / center_y
        dist_from_center = torch.sqrt(dx**2 + dy**2)
        
        # Create center bump (Gaussian)
        center_mask = torch.exp(-(dist_from_center**2) / 0.5)
        
        # Create vignette (stronger near edges)
        # Using distance from center, inverted
        vignette_mask = torch.clamp(dist_from_center - 0.3, 0, 1)
        
        # Combine: center bump + edge vignette
        self.perturbation_mask = center_mask + vignette_mask * 0.5
        
        # Normalize to [0, 1]
        self.perturbation_mask = self.perturbation_mask / self.perturbation_mask.max()
    
    def _create_buttons(self):
        """Create UI buttons"""
        button_width = 180
        button_height = 40
        spacing = 10
        start_x = self.window_width - button_width - 20
        start_y = 20
        
        buttons = [
            {
                "rect": pygame.Rect(start_x, start_y, button_width, button_height),
                "text": "Randomize Image (I)",
                "action": "randomize_image",
                "hovered": False
            },
            {
                "rect": pygame.Rect(start_x, start_y + (button_height + spacing), button_width, button_height),
                "text": "Randomize Kernels (K)",
                "action": "randomize_kernel",
                "hovered": False
            }
        ]
        return buttons
    
    def _create_slider(self):
        """Create drift control sliders"""
        slider_width = 180
        slider_height = 20
        slider_x = self.window_width - slider_width - 20
        slider_y = 140
        spacing = 60
        
        sliders = [
            {
                "rect": pygame.Rect(slider_x, slider_y, slider_width, slider_height),
                "handle_x": slider_x + int(slider_width * (self.drift_scale / self.max_drift_scale)),
                "dragging": False,
                "label": "Drift Scale (Heart)",
                "value_attr": "drift_scale",
                "max_attr": "max_drift_scale",
                "target_obj": self
            },
            {
                "rect": pygame.Rect(slider_x, slider_y + spacing, slider_width, slider_height),
                "handle_x": slider_x + int(slider_width * (self.noise_amount / self.max_noise)),
                "dragging": False,
                "label": "Perturbation",
                "value_attr": "noise_amount",
                "max_attr": "max_noise",
                "target_obj": self
            },
            {
                "rect": pygame.Rect(slider_x, slider_y + spacing * 2, slider_width, slider_height),
                "handle_x": slider_x + int(slider_width * (self.residual_alpha / self.max_residual)),
                "dragging": False,
                "label": "Residual Mix",
                "value_attr": "residual_alpha",
                "max_attr": "max_residual",
                "target_obj": self
            }
        ]
        
        return sliders
    
    def apply_image_noise(self):
        """Add random noise with center + vignette pattern"""
        if self.noise_amount > 0:
            # Generate random noise for all channels
            noise = torch.randn_like(self.image) * self.noise_amount
            
            # Apply mask to focus noise in center and edges
            masked_noise = noise * self.perturbation_mask.unsqueeze(0)
            
            self.image = self.image + masked_noise
            # Clamp to reasonable range
            self.image = torch.clamp(self.image, -1, 1)
    
    def apply_mouse_perturbation(self):
        """Apply localized perturbation where mouse is pressed"""
        if not self.mouse_pressed:
            return
        
        # Calculate how long mouse has been pressed (strength)
        press_duration = pygame.time.get_ticks() - self.mouse_press_start
        strength = min(press_duration / 1000.0, 5.0)  # Max 5 seconds for full strength
        
        # Convert mouse position to image coordinates
        display_size = self.img_width  # Image is full size, no scaling
        mouse_x, mouse_y = self.mouse_pos
        
        # Check if mouse is over the image display area
        img_rect = pygame.Rect(self.display_offset[0], self.display_offset[1], display_size, display_size)
        if not img_rect.collidepoint(mouse_x, mouse_y):
            return
        
        # Convert to image coordinates
        rel_x = (mouse_x - self.display_offset[0]) / display_size
        rel_y = (mouse_y - self.display_offset[1]) / display_size
        
        img_x = int(rel_x * self.img_width)
        img_y = int(rel_y * self.img_height)
        
        # Create a localized perturbation with true zero outside radius
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.img_height, dtype=torch.float32),
            torch.arange(self.img_width, dtype=torch.float32),
            indexing='ij'
        )
        
        # Calculate distance from click point (with wrapping for toroidal topology)
        dx = torch.abs(x_coords - img_x)
        dy = torch.abs(y_coords - img_y)
        
        # Account for wrapping
        dx = torch.min(dx, self.img_width - dx)
        dy = torch.min(dy, self.img_height - dy)
        
        dist = torch.sqrt(dx**2 + dy**2)
        
        # Create Gaussian falloff with hard cutoff at 2*radius
        mask = torch.exp(-(dist**2) / (2 * (self.perturb_radius**2)))
        mask = torch.where(dist <= self.perturb_radius * 2, mask, torch.zeros_like(mask))
        
        # Only generate noise where mask is non-zero (more efficient and truly localized)
        noise = torch.randn(3, self.img_height, self.img_width) * strength * 0.05
        perturbation = noise * mask.unsqueeze(0)
        
        self.image = self.image + perturbation
    
    def image_to_surface(self, img_tensor):
        """Convert torch tensor to pygame surface using efficient blit_array"""
        # Min-max normalization ONLY for display (not part of dynamics)
        img_display = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-6)
        
        # Clamp and convert to [0, 255]
        img_display = torch.clamp(img_display, 0, 1)
        img_np = (img_display.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Blit array directly to surface (swapaxes for pygame's x,y convention)
        pygame.surfarray.blit_array(self.display_surface, img_np.swapaxes(0, 1))
        
        return self.display_surface
    
    def toggle_fullscreen(self):
        """Toggle between windowed and fullscreen mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # Get screen resolution
            info = pygame.display.Info()
            screen_width = info.current_w
            screen_height = info.current_h
            
            # Set fullscreen
            self.screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
            
            # Resize image to match screen resolution
            # Preserve aspect ratio by using smaller dimension
            target_size = min(screen_width, screen_height)
            self.fullscreen_img_size = target_size - 2 * self.conv_processor.padding
            
            # Save old image, create new size, scale up the image state
            old_image = self.image
            old_width, old_height = self.img_width, self.img_height
            
            # Update dimensions
            self.img_width = self.fullscreen_img_size
            self.img_height = self.fullscreen_img_size
            
            # Resize image using interpolation
            old_image_batch = old_image.unsqueeze(0)
            resized = F.interpolate(old_image_batch, size=(self.img_height, self.img_width), 
                                   mode='bilinear', align_corners=False)
            self.image = resized.squeeze(0)
            
            # Recreate display surface
            self.display_surface = pygame.Surface((self.img_width, self.img_height))
            
            # Recreate perturbation mask for new size
            self._create_perturbation_mask()
            
        else:
            # Return to windowed mode
            self.screen = pygame.display.set_mode(self.original_size)
            
            # Save fullscreen image, restore original size
            old_image = self.image
            
            # Restore original dimensions
            self.img_width = 512 - 2 * self.conv_processor.padding
            self.img_height = 512 - 2 * self.conv_processor.padding
            
            # Resize image back down
            old_image_batch = old_image.unsqueeze(0)
            resized = F.interpolate(old_image_batch, size=(self.img_height, self.img_width), 
                                   mode='bilinear', align_corners=False)
            self.image = resized.squeeze(0)
            
            # Recreate display surface
            self.display_surface = pygame.Surface((self.img_width, self.img_height))
            
            # Recreate perturbation mask for new size
            self._create_perturbation_mask()
    
    def handle_button_click(self, pos):
        """Handle button clicks"""
        for button in self.buttons:
            if button["rect"].collidepoint(pos):
                if button["action"] == "randomize_image":
                    self.image = self._random_image()
                elif button["action"] == "randomize_kernel":
                    self.conv_processor.randomize_kernels()
    
    def handle_slider_drag(self, pos, slider):
        """Handle slider dragging"""
        if slider["dragging"]:
            # Update handle position
            slider["handle_x"] = max(slider["rect"].left, 
                                    min(pos[0], slider["rect"].right))
            # Update value
            slider_pos = (slider["handle_x"] - slider["rect"].left) / slider["rect"].width
            target_obj = slider["target_obj"]
            setattr(target_obj, slider["value_attr"].split('.')[-1], 
                   slider_pos * getattr(target_obj, slider["max_attr"].split('.')[-1]))
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw buttons
        for button in self.buttons:
            color = BUTTON_HOVER if button["hovered"] else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, button["rect"], border_radius=5)
            pygame.draw.rect(self.screen, WHITE, button["rect"], width=2, border_radius=5)
            
            # Draw text
            text_surf = self.font.render(button["text"], True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=button["rect"].center)
            self.screen.blit(text_surf, text_rect)
        
        # Draw sliders
        for slider in self.slider:
            # Label
            label_surf = self.font.render(slider["label"], True, TEXT_COLOR)
            self.screen.blit(label_surf, (slider["rect"].x, slider["rect"].y - 25))
            
            # Slider track
            pygame.draw.rect(self.screen, GRAY, slider["rect"], border_radius=3)
            pygame.draw.rect(self.screen, WHITE, slider["rect"], width=2, border_radius=3)
            
            # Slider handle
            handle_radius = 10
            pygame.draw.circle(self.screen, WHITE, 
                             (int(slider["handle_x"]), slider["rect"].centery), 
                             handle_radius)
            
            # Value display
            target_obj = slider["target_obj"]
            value = getattr(target_obj, slider["value_attr"].split('.')[-1])
            value_text = f"{value:.4f}"
            value_surf = self.font.render(value_text, True, TEXT_COLOR)
            self.screen.blit(value_surf, (slider["rect"].x, slider["rect"].y + 25))
        
        # Instructions
        instructions = [
            "Keys:",
            "I - Randomize Image",
            "K - Randomize Kernels",
            "F - Fullscreen",
            "",
            "Click & Hold:",
            "Perturb Image",
            "",
            "ESC - Quit"
        ]
        y_offset = self.window_height - 200
        for i, line in enumerate(instructions):
            text_surf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (self.window_width - 200, y_offset + i * 25))
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_i:
                        self.image = self._random_image()
                    elif event.key == pygame.K_k:
                        self.conv_processor.randomize_kernels()
                    elif event.key == pygame.K_f:
                        # Toggle fullscreen
                        self.toggle_fullscreen()
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Check sliders
                        slider_clicked = False
                        for slider in self.slider:
                            if slider["rect"].collidepoint(event.pos):
                                slider["dragging"] = True
                                self.handle_slider_drag(event.pos, slider)
                                slider_clicked = True
                                break
                        
                        # Check buttons if no slider was clicked
                        if not slider_clicked:
                            button_clicked = False
                            for button in self.buttons:
                                if button["rect"].collidepoint(event.pos):
                                    self.handle_button_click(event.pos)
                                    button_clicked = True
                                    break
                            
                            # If neither slider nor button clicked, start mouse perturbation
                            if not button_clicked:
                                self.mouse_pressed = True
                                self.mouse_press_start = pygame.time.get_ticks()
                                self.mouse_pos = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        for slider in self.slider:
                            slider["dragging"] = False
                        self.mouse_pressed = False
                
                elif event.type == pygame.MOUSEMOTION:
                    # Update button hover states
                    for button in self.buttons:
                        button["hovered"] = button["rect"].collidepoint(event.pos)
                    
                    # Handle slider dragging
                    for slider in self.slider:
                        if slider["dragging"]:
                            self.handle_slider_drag(event.pos, slider)
                    
                    # Update mouse position for perturbation
                    if self.mouse_pressed:
                        self.mouse_pos = event.pos
            
            # Get heart signal and apply kernel drift
            heart_signal = self.heart.beat()
            
            # Change drift direction (random walk on sphere with momentum)
            # Use a constant change amount for smooth evolution
            self.conv_processor.change_drift(change_amount=0.01)
            
            # Apply drift using heart signal
            self.conv_processor.apply_drift(heart_signal * self.drift_scale)
            
            # Apply image noise
            self.apply_image_noise()
            
            # Apply mouse perturbation
            self.apply_mouse_perturbation()
            
            # Apply convolution
            self.image = self.conv_processor.apply(self.image, self.residual_alpha)
            
            # Render
            self.screen.fill(BLACK)
            
            # Draw image
            surface = self.image_to_surface(self.image)
            
            if self.fullscreen:
                # Center image in fullscreen
                screen_width = self.screen.get_width()
                screen_height = self.screen.get_height()
                x = (screen_width - self.img_width) // 2
                y = (screen_height - self.img_height) // 2
                self.screen.blit(surface, (x, y))
            else:
                # Normal windowed mode
                self.screen.blit(surface, (20, 20))
                # Draw UI only in windowed mode
                self.draw_ui()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Use default Seraphim (multi-layer) soul with 60 BPM heart rate
    app = ConvolutionArt()
    
    # Or customize the configuration:
    # app = ConvolutionArt(bpm=80)  # Faster heart rate
    # app = ConvolutionArt(conv_processor=Imp(kernel_size=9), bpm=45)
    # app = ConvolutionArt(conv_processor=Seraphim(kernel_size=5, num_layers=3), bpm=120)
    
    app.run()
