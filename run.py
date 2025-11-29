from main import ConvolutionArt
from soul import Soul, Seraphim, Imp, Cherubim, Ophanim, Nephilim, Thrones, Dominions, Powers

seraphim = Seraphim(kernel_size=5, num_layers=10, drift_magnitude=0.02, momentum=0.7)

ophanim = Ophanim(kernel_size=5, dilations=(1, 2, 4, 8), drift_magnitude=0.015, momentum=0.7)

cherubim = Cherubim(kernel_size=5, latent_channels=16, drift_magnitude=0.18, momentum=0.6)

nephilim = Nephilim(kernel_size=7, num_kernels=4, drift_magnitude=0.002, momentum=0.75,
                    turbulence_scale=0.3, chaos_speed=0.05, zone_frequency=2.0)

thrones = Thrones(kernel_size=5, num_scales=3, drift_magnitude=0.003, momentum=0.7,
                  flow_strength=0.15, scale_blend=0.5, flow_frequency=1.5) # One trick pony, flows dominate

dominions = Dominions(kernel_size=7, num_orientations=8, num_frequencies=2, 
                      drift_magnitude=0.0025, momentum=0.75, gradient_sensitivity=0.6) # One trick pony, gradients dominate

powers = Powers(kernel_size=5, num_layers=6, drift_magnitude=0.015, momentum=0.7,
                attention_strength=0.5, feature_mix=0.6)

soul = powers

app = ConvolutionArt(conv_processor=soul, bpm=60)
app.run()