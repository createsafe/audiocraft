import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch import nn
import torchaudio

import librosa
from BeatNet.BeatNet import BeatNet

def t_linspace(start: float, stop: float, size: int, endpoint: bool=False) -> torch.Tensor:
    """
    Torch linspace that behaves more like numpy.linspace
    """
    if not endpoint:
        step = (stop - start) / size
        stop -= step
    return torch.linspace(start=start, end=stop, steps=size)

def impulse2sawtooth(signal: torch.Tensor) -> torch.Tensor:
    """
    Generate sawtooth from impulse train.

    Note, sawtooth is aliased.
    """
    i = torch.where(signal==1)[0]
    result = torch.zeros_like(signal)
    result[i] = 1.0

    for n in range(len(i)-1):
        start = i[n]+1
        end = i[n+1]
        result[start:end] = t_linspace(0, 1, end-start, False)

    return result

def beats2sawtooth(wav, sample_rate, hop_size, beat_times, beat_positions):
    BEAT = 0
    DOWNBEAT = 1
    NUM_BEAT_CLASSES = 2

    num_samples = wav.shape[-1]
    duration = num_samples/sample_rate
    hop_times = torch.arange(0, duration, step=hop_size/sample_rate)
    num_frames = len(hop_times)
    frames = torch.zeros(size=(NUM_BEAT_CLASSES, num_frames))

    beat_frames = list()
    downbeat_frames = list()

    for n in range(len(hop_times[:-1])):
        # find frames in which beats fall
        if any([t >= hop_times[n] and t < hop_times[n+1] for t in beat_times]):
            frames[BEAT, n] = 1
            beat_frames.append(n)
        # find frames in which downbeats fall
        if any([t >= hop_times[n] and t < hop_times[n+1] and p == 1 for t, p in zip(beat_times, beat_positions)]):
            frames[DOWNBEAT, n] = 1
            downbeat_frames.append(n)

    for n in range(NUM_BEAT_CLASSES):
        frames[n, :] = impulse2sawtooth(frames[n, :])

    return frames

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
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)

    def forward(self, wav: torch.Tensor):
        T = wav.shape[-1]

        if T < 4096:
            frames = torch.zeros((2, 1))
        else:
            beats = self.estimator.process_offline(wav, self.sample_rate)
            beat_times = beats[:, 0]
            beat_positions = beats[:, 1]
            
            frames = beats2sawtooth(wav, 
                                    sample_rate=self.sample_rate, 
                                    hop_size=self.hop_size,
                                    beat_times=beat_times,
                                    beat_positions=beat_positions)
        
        return frames

def main():
    import os 

    file = "audio/80bpm.wav"
    audio, sample_rate = librosa.load(file, mono=True)
    extractor = BeatExtractor(sample_rate=sample_rate, hop_size=512)
    frames = extractor.forward(audio)
    hop_times = np.linspace(0, len(audio)/sample_rate, frames.shape[-1])

    t = np.linspace(0, len(audio)/sample_rate, len(audio), False)
    beat_idxs = np.where(frames[0, :] == 1)
    downbeat_idxs = np.where(frames[1, :] == 1)

    # plot 
    fig, axs = plt.subplots(4, 1, sharex=True, tight_layout=True)
    axs[0].plot(t, audio)
    axs[0].set_title(f"audio from {file}")
    axs[1].vlines(hop_times[beat_idxs], ymin=-1, ymax=1)
    axs[1].set_title("beats found")
    axs[2].plot(hop_times, frames[0, :])
    axs[2].set_title("condition dim 1")
    axs[3].plot(hop_times, frames[1, :])
    axs[3].set_title("condition dim 2")
    axs[3].set_xlabel("frame")
    plt.xlim((20, 40))
    plt.savefig(fname=f"{os.path.basename(file)}.png", dpi=300)

if __name__ == "__main__":
    main()