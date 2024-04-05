import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn

import librosa
from BeatNet.BeatNet import BeatNet

estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

file = "path/to/audio.wav"
audio, sample_rate = librosa.load(file, mono=True)
beats = estimator.process(file)
beats = {
    "times": beats[:, 0],
    "pos": beats[:, 1]
}

duration = len(audio)/sample_rate

hop_size = 512
hop_times = np.arange(start=0, stop=duration, step=hop_size/sample_rate)
frames = np.zeros_like(hop_times)

for n, _ in enumerate(hop_times[:-1]):
    if any([t >= hop_times[n] and t < hop_times[n+1] for t in beats['times']]):
        frames[n] = 1

frames_with_beats = np.where(frames)[0]
for n, _ in enumerate(frames_with_beats[:-1]):
    start = frames_with_beats[n]+1
    end = frames_with_beats[n+1]
    frames[start:end] = np.linspace(0, 1, end-start, False)


t = np.linspace(0, len(audio)/sample_rate, len(audio), False)
plt.plot(t, audio)
# plt.vlines(x=beats['times'], ymin=-1, ymax=1, colors='r')
plt.plot(hop_times, frames)
plt.show()