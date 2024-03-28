import os
import pprint
import glob

import pandas as pd
import numpy as np

import soundfile as sf

import demucs
import demucs.separate
import demucs.pretrained
import demucs.apply
from demucs.audio import convert_audio

def separate_vocals(audio_path: str, 
                    model: demucs.apply.BagOfModels,
                    device: str="cuda") -> tuple[np.ndarray, np.ndarray, int]:
    """
    separate vocals from instrumental in music. 

    inputs:
        `audio_path`: path/to/audio.wav
        `model`: model retrieved from `demucs.pretrained` e.g. `demucs.pretrained.get_model("htdemucs")`
    returns:
        `voice`: voice audio signal
        `background`: backgroud audio signal
        `sample_rate`: audio sample rate for both signals
    """
    wav = demucs.separate.load_track(track=audio_path, audio_channels=model.audio_channels, samplerate=model.samplerate)

    # audio, sample_rate = sf.read(audio_path)
    # wav = convert_audio(wav=torch.from_numpy(audio).T,
    #                     from_samplerate=sample_rate,
    #                     to_samplerate=model.samplerate,
    #                     channels=model.audio_channels)

    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    sources = demucs.apply.apply_model(model=model, 
                                        mix=wav[None],
                                        device=device,
                                        split=True, 
                                        segment=None,
                                        progress=False)[0]
    sources *= ref.std()
    sources += ref.mean()

    voice = np.zeros(sources[0].shape).T
    background = np.zeros_like(voice)
    for n in range(sources.shape[0]):
        if model.sources[n] == "vocals":
            voice = sources[n].numpy().T
        else: 
            background += sources[n].numpy().T

    return voice, background, model.samplerate

def main():
    # path = "audio-training-data/grimes-music-ingest-tracks/.song_ids"
    # tracks = pd.read_csv(path, sep='\t')
    # pprint.pprint(tracks)

    model = demucs.pretrained.get_model("htdemucs")

    voice_dir = os.path.join("audio-training-data", "grimesai0", "voice")
    instrumental_dir = os.path.join("audio-training-data", "grimesai0", "instrumental")
    for dir in [voice_dir, instrumental_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    files = glob.glob(os.path.join("audio-training-data", "grimesai0", "raw", "*.mp3"))
    num_files = len(files)
    files = files[:10]
    for n, filepath in enumerate(files):
        name = os.path.basename(filepath)
        voice_audio, instrumental_audio, sample_rate = separate_vocals(filepath, model)
        sf.write(os.path.join(voice_dir, name), voice_audio, sample_rate)
        sf.write(os.path.join(instrumental_dir, name), instrumental_audio, sample_rate)
        print(f"{n+1}/{num_files}: {name}")

if __name__ == "__main__":
    main()
