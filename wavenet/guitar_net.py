from time import time
import os

from wavenet.utils import make_batch
from wavenet.models import Model, Generator
import scipy

from IPython.display import Audio
from matplotlib import pyplot as plt
import numpy as np

print os.path.join(os.path.curdir, "assets", "voice.wav")
inputs, targets = make_batch('../assets/voice.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1.0

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)

Audio(inputs.reshape(inputs.shape[1]), rate=44100)


for epoch in range(100):
    tic = time()
    model.train(inputs, targets)
    generator = Generator(model)
    toc = time()
    print('Epoch took {} seconds.'.format(toc - tic))
    # Get first sample of input
    input_ = inputs[:, 0:1, 0]
    plt.plot(model.losses)
    plt.savefig(str(epoch) +"_loss.png")
    plt.clf()

    tic = time()
    predictions = generator.run(input_, 10 * 44100)
    print predictions
    print predictions[0]
    toc = time()
    predictions = np.array(predictions[0])
    print predictions
    print predictions.dtype
    print('Generating took {} seconds.'.format(toc - tic))
    scipy.io.wavfile.write(str(epoch) + ".wav", 44100, predictions)
    # Audio(predictions, rate=44100)






