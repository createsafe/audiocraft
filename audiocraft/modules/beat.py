from enum import Enum
import typing as tp

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

class MetricalUnits(Enum):
    BEAT = 0
    MEASURE = 1

def beats2sawtooth(*, 
                   wav: torch.Tensor=None, 
                   sample_rate: int=None, 
                   hop_size: int=None, 
                   beat_times: tp.Iterable=None, 
                   beat_positions: tp.Iterable=None,
                   num_frames: int=None) -> torch.Tensor:
    """Turn a list of beats into metrical phase matrix.

    Args:
        wav (torch.Tensor, optional): Audio input if available. Defaults to None.
        sample_rate (int, optional): signal sample rate. Defaults to None.
        hop_size (int, optional): Hop size. Defaults to None.
        beat_times (tp.Iterable, optional): List of beat times. Defaults to None.
        beat_positions (tp.Iterable, optional): List of beat positions. Defaults to None.
        num_frames (int, optional): Number of frames. Defaults to None.

    Returns:
        torch.Tensor: ramps going from 0 to 1 synchronous to metrical units of beat and 
                      measure.

    Usage: 
    - use either with `wav` or `num_frames`, but not both!
    """

    assert wav or num_frames, f"Must provide `wav` or `num_frames`."
    if wav and num_frames:
        Warning(f"`wav` and `num_frames` provided; using `wav`.")

    if wav:
        num_samples = wav.shape[-1]
        duration = num_samples/sample_rate
        hop_times = torch.arange(0, duration, step=hop_size/sample_rate)
        num_frames = len(hop_times)
    else:
        duration = num_frames * hop_size / sample_rate
        hop_times = torch.arange(0, duration, step=hop_size/sample_rate)

    frames = torch.zeros(size=(len(MetricalUnits), num_frames))

    for n in range(len(hop_times[:-1])):
        # find frames in which beats fall
        if any([t >= hop_times[n] and t < hop_times[n+1] for t in beat_times]):
            frames[MetricalUnits.BEAT.value, n] = 1
        # find frames in which downbeats fall
        if any([t >= hop_times[n] and t < hop_times[n+1] and p == 1 for t, p in zip(beat_times, beat_positions)]):
            frames[MetricalUnits.MEASURE.value, n] = 1

    for n in range(len(MetricalUnits)):
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
                 hop_size: int = 512, 
                 device = "cpu"):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.device = device
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False, device=device)

    def forward(self, wav: torch.Tensor, num_frames: int):

        num_samples = wav.shape[-1]

        if not num_frames:
            num_frames = int(np.ceil(num_samples / self.hop_size))
        
        beat_feature = torch.zeros(size=(1, num_frames, 2))

        if num_samples < 4096:
            pass
        else:
            beats = self.estimator.get_beats(audio=wav, sample_rate=self.sample_rate)
            # beats = self.estimator.process_offline(wav.cpu().numpy().T, self.sample_rate)
            beat_times = beats[:, 0]
            beat_positions = beats[:, 1]
            
            frames = beats2sawtooth(num_frames=num_frames, 
                                    sample_rate=self.sample_rate, 
                                    hop_size=self.hop_size,
                                    beat_times=beat_times,
                                    beat_positions=beat_positions).T
            
            beat_feature = frames.unsqueeze(0)

        return beat_feature

def main():
    import os 

    file = "audio/80bpm.wav"
    audio, sample_rate = librosa.load(file, mono=True)
    audio = torch.Tensor(audio)
    extractor = BeatExtractor(sample_rate=sample_rate, hop_size=512)
    frames = extractor.forward(audio.unsqueeze(0).unsqueeze(0), sample_rate)

    dur = audio.shape[-1]/sample_rate
    hop_times = np.linspace(0, dur, frames.shape[-1])

    t = np.linspace(0, dur, len(audio), False)
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