from main import ConvolutionArt
from soul import Soul, Seraphim, Imp, Cherubim, Ophanim

seraphim = Seraphim(kernel_size=5, num_layers=10, drift_magnitude=0.02, momentum=0.7)

ophanim = Ophanim(kernel_size=5, dilations=(1, 2, 4, 8), drift_magnitude=0.015, momentum=0.7)

cherubim = Cherubim(kernel_size=5, latent_channels=16, drift_magnitude=0.18, momentum=0.6)

soul = ophanim

app = ConvolutionArt(conv_processor=soul, bpm=60)
app.run()