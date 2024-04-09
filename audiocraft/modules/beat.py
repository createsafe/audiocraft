import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn

import librosa
from BeatNet.BeatNet import BeatNet

class BeatExtractor(nn.Module):
    """Beat extraction and time ramp generation.

        Extract beats using BeatNet[1]. Assign beats on a per-frame basis, 
        where frames with a beat are assigned 1. Between beats is a ramp 
        from 0 to 1 indicating the metrical position within the beat.

        Args:
            sample_rate (int): Sample rate for beat extraction
            hop_size (int): subsampling period for beat resolution
        
        Resource:
        [1] Heydari, Mojtaba, Frank Cwitkowitz, and Zhiyao Duan. “BeatNet: 
        CRNN and Particle Filtering for Online Joint Beat Downbeat and Meter 
        Tracking.” arXiv, August 8, 2021. http://arxiv.org/abs/2108.03576.

    """
    def __init__(self,
                 sample_rate: int,
                 hop_size: int = 512):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    def forward(self, wav: torch.Tensor):
        beats = self.estimator.offline_process(wav.numpy(), self.sample_rate)
        beat_times = beats[:, 0]
        beat_positions = beats[:, 1]
        
        duration = len(wav)/self.sample_rate
        hop_times = np.linspace(0, duration, len(wav))
        num_hops = len(hop_times)
        frames = np.zeros((2, num_hops))

        # find frames that contain beats
        for n in range(len(hop_times[:-1])):
            if any([t >= hop_times[n] and t < hop_times[n+1] for t in beat_times]):
                frames[0, n] = 1
                if beat_positions[n] == 1:
                    frames[1, n] = 1

        frames_with_beats = np.where(frames)[0]
        for n, _ in enumerate(frames_with_beats[:-1]):
            start = frames_with_beats[n]+1
            end = frames_with_beats[n+1]
            frames[0, start:end] = np.linspace(0, 1, end-start, False)

        return torch.from_numpy(frames)

if __name__ == "__main__":
    file = "path/to/audio.wav"
    audio, sample_rate = librosa.load(file, mono=True)
    extractor = BeatExtractor(sample_rate=sample_rate, hop_size=512)
    frames = extractor.forward(torch.from_numpy(audio))
    hop_times = np.linspace(0, len(audio)/sample_rate, len(frames))

    t = np.linspace(0, len(audio)/sample_rate, len(audio), False)
    plt.plot(t, audio)
    # plt.vlines(x=beats['times'], ymin=-1, ymax=1, colors='r')
    plt.plot(hop_times, frames)
    plt.show()