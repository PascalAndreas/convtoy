from main import ConvolutionArt
from soul import Soul, Seraphim, Imp, Cherubim, Ophanim, Nephilim, Thrones, Dominions, Powers, Metatron, Leviathan, Abaddon, Pandemonium, Janus, Azazel, Eidolon

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
                attention_strength=0.5, feature_mix=0.6) # Mid

metatron = Metatron(kernel_size=5, latent_channels=10, drift_magnitude=0.0025,
                    momentum=0.82, cross_coupling=0.4, phase_twist=0.5,
                    memory_decay=0.2, shear_strength=0.32, resonance=0.6) # Like 6/10, but more interesting

leviathan = Leviathan(kernel_size=5, latent_channels=12, drift_magnitude=0.002,
                      momentum=0.8, skip_blend=0.45, ripple_strength=0.28,
                      tide_strength=0.35, depth_mix=0.6, bottleneck_dilation=2) # Rainbow cloud textures

abaddon = Abaddon(kernel_size=5, latent_channels=14, drift_magnitude=0.028,
                  momentum=0.78, warp_strength=0.35, warp_freq=3.3,
                  glitch_strength=0.22, glitch_density=0.35, chaos_rate=0.4,
                  depth_mix=0.55) # Absolute chaos engine - Really fucking cool! Looks like brish strokes! Residual needs to be low, around 0.2.

pandemonium = Pandemonium(kernel_size=5, latent_channels=12, num_regions=8,
                          drift_magnitude=0.02, momentum=0.78, warp_strength=0.3,
                          warp_speed=0.45, glitch_strength=0.18, region_sharpness=2.5,
                          depth_mix=0.55, turbulence=0.4) # Dynamic chaotic regions, low residual recommended - mid

janus = Janus(kernel_size=5, latent_channels=10, drift_magnitude=0.0018,
              momentum=0.82, symmetry_strength=0.55, anchor_focus=0.6,
              feature_sharpness=1.2, smile_curve=0.35) # Face pareidolia coaxer - mid

azazel = Azazel(kernel_size=5, latent_channels=14, num_regions=4,
                drift_magnitude=0.003, momentum=0.78, warp_strength=0.32,
                warp_speed=0.5, region_sharpness=2.6, depth_mix=0.6,
                turbulence=0.45, glitch_strength=0.2) # Flowfield chaos, low residual - mid

eidolon = Eidolon(kernel_size=5, latent_channels=12, drift_magnitude=0.0016,
                  momentum=0.82, symmetry_strength=0.6, anchor_focus=0.7,
                  feature_sharpness=1.1, smile_curve=0.3, prior_weight=0.5) # Face-seeking phantom, low residual

soul = abaddon

app = ConvolutionArt(conv_processor=soul, bpm=60)
app.run()
