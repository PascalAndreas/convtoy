from main import ConvolutionArt
from soul import Soul, Seraphim, Imp, Cherubim, Ophanim

seraphim = Seraphim(kernel_size=7, num_layers=5, drift_magnitude=0.02, momentum=0.7)

ophanim = Ophanim(kernel_size=5, dilations=(8, 2, 4, 2, 1), drift_magnitude=0.015, momentum=0.7)

cherubim = Cherubim(kernel_size=5, latent_channels=16, drift_magnitude=0.18, momentum=0.6)

soul = seraphim

app = ConvolutionArt(conv_processor=soul, bpm=60)
app.run()